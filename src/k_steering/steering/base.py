from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import PushToHubMixin
import pickle
from collections import defaultdict
from huggingface_hub import hf_hub_download

from src.k_steering.steering.dataset import TaskDataset
from src.k_steering.steering.config import SteeringConfig, TrainerConfig


class ActivationSteering(ABC, PushToHubMixin):
    """
    Base class for activation steering methods

    Provides common functionality for:
    - Model loading
    - Caching hidden states
    - Save/load operations
    - Basic generation interface

    Subclasses should implement:
    - build_steering_trainer()
    - _apply_steering()
    """

    def __init__(
        self,
        model_name: str,
        steering_config: Optional[SteeringConfig] = None,
        trainer_config: Optional[TrainerConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize activation steering base

        Args:
            model_name (str): HuggingFace model name or path
            steering_config (SteeringConfig): Configuration for steering behavior
            trainer_config (TrainerConfig): Configuration for training classifier
            device (str): Device to load model on (auto-detected if None)
        """
        self.model_name = model_name
        self.steering_config = steering_config or SteeringConfig()
        self.trainer_config = trainer_config or TrainerConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.model, self.tokenizer = self._load_hf_model(model_name)

        # State variables (populated during fit)
        self.dataset = None
        self.unique_labels = None
        self.eval_prompts = None
        self.cache = None
        self._is_fitted = False

        # Steering components (subclass-specific)
        self.steering_vectors = {}
        self.k_clf = None

    def _load_hf_model(
        self, model_name: str
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load HuggingFace model and tokenizer

        Args:
            model_name (str): Huggingface Model identifier

        Returns:
            Tuple of (model, tokenizer)
        """
        print(f"Loading Model: {model_name} from HuggingFace")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            _attn_implementation="eager",
            output_hidden_states=True,
            device_map="auto" if self.device == "cuda" else None,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        return model, tokenizer

    def _load_task(self, task_name: str) -> Tuple[Any, List[str], List[str]]:
        """
        Load predefined task dataset

        Args:
            task_name (str): Name of Predefined task to load

        Returns:
            Tuple of (dataset, unique_labels, eval_prompts)
        """
        # TODO: Implement task loading from registry
        raise NotImplementedError(
            f"Task loading for '{task_name}' not implemented. "
            "Override this method or provide custom dataset."
        )

    def _extract_labels(self, dataset: Any) -> List[str]:
        """
        Extract unique labels from dataset

        Args:
            dataset (Any): Dataset object

        Returns:
            List of unique label strings
        """
        # Default implementation assumes dataset has 'label' field
        if hasattr(dataset, "__iter__"):
            labels = [item.get("label") for item in dataset if "label" in item]
            return list(set(labels))
        raise NotImplementedError("Override this method for custom dataset structure")

    def get_hidden_cache(
        self,
        prompts: Dict[str, List[str]],  # e.g. {"train": [...], "eval": [...]}
        batch_size: int = 64,
        return_attention_mask: bool = True,
    ) -> Dict[str, Dict[Union[int, str], torch.Tensor]]:
        """
        Get cached hidden states for multiple named prompt splits.

        Args:
            prompts (Dict): Dict mapping split name -> list of prompts
            batch_size (int): Batch size for processing
            return_attention_mask (bool): Whether to include attention masks

        Returns:
            {
            split_name: {
                layer_idx: Tensor [B, T, D],
                ...
                "attention_mask": Tensor [B, T]
            }
            }
        """
        num_layers = self.model.config.num_hidden_layers
        all_caches: Dict[str, Dict[Union[int, str], torch.Tensor]] = {}

        with torch.no_grad():
            for split_name, split_prompts in prompts.items():
                cache = {i: [] for i in range(num_layers + 1)}
                attention_masks = [] if return_attention_mask else None

                for i in range(0, len(split_prompts), batch_size):
                    batch = split_prompts[i : i + batch_size]

                    inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(self.device)

                    if return_attention_mask:
                        attention_masks.append(inputs["attention_mask"])

                    outputs = self.model(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                    for layer_idx, hidden in enumerate(outputs.hidden_states):
                        cache[layer_idx].append(hidden.cpu())

                # concatenate batches
                cache = {k: torch.cat(v, dim=0) for k, v in cache.items()}

                if return_attention_mask:
                    cache["attention_mask"] = torch.cat(attention_masks, dim=0)

                all_caches[split_name] = cache

        return all_caches

    def get_layer_cache(
        self,
        split_name: str,
        layer_idx: int,
        prompts: Optional[Dict[str, List[str]]] = None,
        batch_size: int = 64,
        use_cached: bool = True,
    ) -> torch.Tensor:
        """
        Extract the last non-padding token representation for a given
        layer and named split.
        
        Args:
            split_name (str): Split Name for prompt type
            layer_idx (int) : Layer ID for layer wise cache
            prompts (Dict) : Dict of Prompts with Named Splits (only when cache is not defined)
            batch_size (int) : Batch Size for generating cache
            use_cache (bool): Boolean for using existing cache or not

        Returns:
            Tensor of shape [num_prompts, hidden_dim]
        """
        # ---------- ensure cache for split exists ----------
        if (
            not use_cached
            or self.cache is None
            or split_name not in self.cache
            or "attention_mask" not in self.cache[split_name]
        ):
            if prompts is None or split_name not in prompts:
                raise ValueError(
                    f"Must provide prompts for split '{split_name}' "
                    "if cache is not available"
                )

            # compute cache only for this split
            self.cache = self.cache or {}
            self.cache.update(
                self.get_hidden_cache(
                    {split_name: prompts[split_name]},
                    batch_size=batch_size,
                    return_attention_mask=True,
                )
            )

        split_cache = self.cache[split_name]
        hidden = split_cache[layer_idx] 
        attention_mask = split_cache["attention_mask"]   # [B, T]

        # last non-padding token index
        lengths = attention_mask.sum(dim=1) - 1

        # collect last-token representations
        vectors = [hidden[i, idx] for i, idx in enumerate(lengths)]

        return torch.stack(vectors, dim=0)

    @abstractmethod
    def build_steering_trainer(
        self, cache: Dict[str, torch.Tensor], eval: bool = False
    ):
        """
        Build steering classifier/vectors from cached activations

        Args:
            cache (dict): Cached hidden states
            eval (bool): Whether to build evaluation classifier

        Returns:
            Classifier/steering vectors (subclass-specific)
        """
        pass

    @abstractmethod
    def _apply_steering(
        self, hidden_states: torch.Tensor, layer_idx: int, steering_strength: float
    ) -> torch.Tensor:
        """
        Apply steering to hidden states at a specific layer

        Args:
            hidden_states (torch.Tensor): Original hidden states
            layer_idx (int): Current layer index
            steering_strength (float): Strength of steering

        Returns:
            torch.Tensor: Modified hidden states
        """
        pass

    def fit(
        self,
        task: Optional[str] = None,
        dataset: Optional[Any] = None,
        eval_prompts: Optional[List[str]] = None,
        batch_size: int = 64,
    ) -> "ActivationSteering":
        """
        Fit the steering model on a task or dataset

        Args:
            task (str): Name of predefined task
            dataset (Any): Custom dataset
            eval_prompts (list): Optional evaluation prompts
            batch_size (int): Batch size for caching

        Returns:
            self for method chaining
        """
        # Load data
        if task is not None:
            self.dataset, self.unique_labels, self.eval_prompts = self._load_task(task)
        elif dataset is not None:
            self.dataset = dataset
            self.unique_labels = self._extract_labels(dataset)
            self.eval_prompts = eval_prompts or []
        else:
            raise ValueError("Either 'task' or 'dataset' must be provided")
        
        self.tone2idx = {t: i for i, t in enumerate(self.unique_labels)}

        # Prepare prompts
        train_prompts = self._get_prompts_from_dataset(self.dataset)
        self.n_train_prompts = len(train_prompts)
        all_prompts = {
                            "train": train_prompts,
                            "eval": self.eval_prompts,
                        }

        # Cache hidden states
        print(f"Caching hidden states for {all_prompts.keys()} prompt splits...")
        self.cache = self.get_hidden_cache(all_prompts, batch_size=batch_size)

        # Build classifier/steering vectors
        print("Building Steering Trainer...")
        self.build_steering_trainer(eval=False)
        self._is_fitted = True

        print("Training complete!")
        return self

    def _get_prompts_from_dataset(self, dataset: Any) -> List[str]:
        """
        Extract prompts from dataset

        Args:
            dataset (Any): Dataset object

        Returns:
            List of prompt strings
        """
        if hasattr(dataset, "__iter__"):
            prompts = [item.get("prompt", item.get("text", "")) for item in dataset]
            return [p for p in prompts if p]
        raise NotImplementedError("Override for custom dataset structure")

    def get_steered_output(
        self,
        input_prompts: List[str],
        steering_strength: Optional[float] = None,
        layers: Optional[List[int]] = None,
        target_labels: Optional[List[str]] = None,
        avoid_labels: Optional[List[str]] = None,
        layer_strengths: Optional[Dict[int, float]] = None,
        max_new_tokens: int = 100,
        return_dict: bool = False,
        **generation_kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate steered output for input prompt

        Args:
            input_prompt (list): Input text
            steering_strength (float): Override default strength
            layers (list): Override target layers
            target_labels (list): Labels for steering behaviour towards
            avoid_labels (list): Labels for steering behaviour away from 
            layer_strengths (dict): Layer-specific strengths
            max_new_tokens (int): Maximum tokens to generate
            return_dict (bool): Return full output dictionary
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated text or output dictionary
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model must be fitted before generation. Call fit() first."
            )

        # Get effective steering parameters
        self.steering_strength = steering_strength or self.steering_config.steering_strength
        self.target_layers = layers or self.steering_config.steer_layers
        self.layer_strengths = layer_strengths or self.steering_config.layer_strengths
        self.target_lbls = target_labels
        self.avoid_lbls = avoid_labels

        # Prepare generation
        self.gen_kwargs = self._prepare_generation_kwargs(
            max_new_tokens=max_new_tokens, **generation_kwargs
        )

        # Subclass-specific steering implementation
        output = self._generate_with_steering(
            input_prompts=input_prompts,
            steering_strength = self.steering_strength,
            target_labels=self.target_lbls,
            avoid_labels=self.avoid_lbls,
            target_layers=self.target_layers,
            layer_strengths=self.layer_strengths,
            generation_kwargs=self.gen_kwargs
        )

        if return_dict:
            return output
        return output.get("text", output)

    def _prepare_generation_kwargs(
        self,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare generation keyword arguments

        Args:
            max_new_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling parameter
            do_sample (bool): Whether to use sampling
            **kwargs: Additional generation parameters

        Returns:
            Dictionary of generation parameters
        """
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        gen_kwargs.update(kwargs)
        return gen_kwargs

    def _generate_with_steering(
        self,
        input_prompt: str,
        steering_strength: float,
        target_labels: List[str],
        avoid_labels: Optional[List[str]],
        target_layers: Optional[List[int]],
        layer_strengths: Dict[int, float],
        generation_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate text with steering applied (base implementation)

        Args:
            input_prompt (str): Input text
            steering_strength (float): Global steering strength
            target_labels (list): Labels for steering behaviour towards
            avoid_labels (list): Labels for steering behaviour away from 
            target_layers (list): Layers to apply steering
            layer_strengths (dict): Layer-specific strengths
            generation_kwargs: Generation parameters

        Returns:
            Dictionary with generation results
        """
        # Basic implementation - subclasses can override
        inputs = self.tokenizer(input_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return {
            "text": generated_text,
            "input_prompt": input_prompt,
            "steering_strength": steering_strength,
            "target_layers": target_layers,
        }

    def save(
        self,
        model_path: Union[str, Path],
        filename: str,
        *,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        commit_message: str = "Add steering model artifacts",
        private: Optional[bool] = None,
    ) -> None:
        """
        Save steering model to Disk or Huggingface

        Args:
            model_path (str): Directory to save
            filename (str): Base filename
            push_to_hub (bool): Boolean flag if model is to be saved on Hugginface
            repo_id (str) : Hugginface repository id for model 
            commit_message (str): Commit Message for Hugginface
            private (bool): Boolean flag for pushing to a private huggingface repository
        """
        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)

        # Save steering config
        config_path = model_path / f"{filename}_steering_config.json"
        with open(config_path, "w") as f:
            json.dump(self.steering_config.to_dict(), f, indent=2)

        # Save trainer config
        config_path = model_path / f"{filename}_trainer_config.json"
        with open(config_path, "w") as f:
            json.dump(self.trainer_config.to_dict(), f, indent=2)

        # Save classifier/steering vectors
        if self.k_clf is not None:
            clf_path = model_path / f"{filename}_classifier.pkl"
            with open(clf_path, "wb") as f:
                pickle.dump(self.k_clf, f)

        if self.steering_vectors is not None:
            vec_path = model_path / f"{filename}_vectors.pt"
            torch.save(self.steering_vectors, vec_path)

        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "unique_labels": self.unique_labels,
            "is_fitted": self._is_fitted,
            "class_name": self.__class__.__name__,
        }
        metadata_path = model_path / f"{filename}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to {model_path / filename}")

        if push_to_hub:
            self.push_to_hub(
                repo_id=repo_id,
                commit_message=commit_message,
                private=private,
                local_dir=str(model_path),
            )

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        filename: str = "activation_steering",
        **kwargs,
    ):
        self.save(
            model_path=save_directory,
            filename=filename,
            push_to_hub=False,  # important: avoid recursion
        )
        
    def _get_repo_url(self, repo_id: str) -> str:
        return f"https://huggingface.co/{repo_id}"

    @classmethod
    def load(
        cls,
        model_path: Union[str, Path],
        filename: str,
        *,
        repo_id: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> "ActivationSteering":
        """
        Load steering model from local disk or Hugging Face Hub.

        Args:
            model_path (str): Local directory OR cache dir for HF download
            filename (str): Base filename
            repo_id (str): Hugging Face repo id (if loading from hub)
            revision (str): Branch / tag / commit
        """
        # ---------- resolve files ----------
        if repo_id is not None:
            # Download artifacts from HF into cache
            def hf(path: str) -> Path:
                return Path(
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=path,
                        revision=revision,
                        cache_dir=model_path
                    )
                )

            metadata_path = hf(f"{filename}_metadata.json")
            steering_config_path = hf(f"{filename}_steering_config.json")
            trainer_config_path = hf(f"{filename}_trainer_config.json")
            try:
                clf_path = hf(f"{filename}_classifier.pkl")
            except Exception as e:
                pass
            try:
                vec_path = hf(f"{filename}_vectors.pt")
            except Exception as e:
                pass

        else:
            model_path = Path(model_path)
            metadata_path = model_path / f"{filename}_metadata.json"
            steering_config_path = model_path / f"{filename}_steering_config.json"
            trainer_config_path = model_path / f"{filename}_trainer_config.json"
            clf_path = model_path / f"{filename}_classifier.pkl"
            vec_path = model_path / f"{filename}_vectors.pt"

        # ---------- load metadata ----------
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # ---------- load steering config ----------
        with open(steering_config_path, "r") as f:
            steering_config_dict = json.load(f)
        steering_config = SteeringConfig.from_dict(steering_config_dict)

        # ---------- load trainer config ----------
        with open(trainer_config_path, "r") as f:
            trainer_config_dict = json.load(f)
        trainer_config = TrainerConfig.from_dict(trainer_config_dict)

        # ---------- initialize instance ----------
        instance = cls(metadata["model_name"], steering_config, trainer_config)
        instance.unique_labels = metadata["unique_labels"]
        instance._is_fitted = metadata["is_fitted"]
        instance.tone2idx = {t: i for i, t in enumerate(metadata['unique_labels'])}

        # ---------- load classifier ----------
        # if clf_path.exists():
        try:
            with open(clf_path, "rb") as f:
                instance.k_clf = pickle.load(f)
        except UnboundLocalError as e:
            pass
    
        # ---------- load steering vectors ----------
        # if vec_path.exists():
        try:
            instance.steering_vectors = torch.load(vec_path, map_location="cpu")
        except UnboundLocalError as e:
            pass

        print(
            f"Model loaded from "
            f"{repo_id if repo_id is not None else metadata_path.parent}"
        )
        return instance

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"fitted={self._is_fitted}, "
            f"steering_strength={self.config.steering_strength})"
        )
        

