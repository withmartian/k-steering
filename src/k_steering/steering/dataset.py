


from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd


@dataclass(frozen=True)
class DatasetSchema:
    prompt_column: str
    category_columns: List[str]

    @property
    def all_columns(self) -> List[str]:
        return [self.prompt_column, *self.category_columns]
    

class TaskDataset:
    """
    Base Task Dataset class for external integrations.

    Required schema:
        prompt
        category_1_response
        category_2_response
        ...
        category_k_response
    """

    def __init__(
        self,
        data: pd.DataFrame,
        schema: DatasetSchema,
        strict: bool = True,
    ):
        self.data = data.reset_index(drop=True)
        self.schema = schema
        self.strict = strict

        self._validate()

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        schema: DatasetSchema,
        strict: bool = True,
    ) -> "TaskDataset":
        return cls(
            data=df,
            schema=schema,
            strict=strict,
        ).to_labeled_examples()

    @classmethod
    def from_huggingface(
        cls,
        repo_id: str,
        *,
        split: str = "train",
        schema: DatasetSchema,
        strict: bool = True,
        cast_to_str: bool = True,
        streaming: bool = False,
    ) -> "TaskDataset":
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "HuggingFace datasets is required for `from_huggingface`.\n"
                "Install with: pip install datasets"
            ) from e

        hf_ds = load_dataset(
            repo_id,
            split=split,
            streaming=streaming,
        )

        if streaming:
            raise NotImplementedError(
                "Streaming mode is not yet supported. "
                "Convert to pandas-backed dataset first."
            )

        cls._assert_hf_schema(hf_ds, schema)

        df = hf_ds.to_pandas()

        if cast_to_str:
            for col in schema.all_columns:
                df[col] = df[col].astype(str)

        return cls(
            data=df,
            schema=schema,
            strict=strict,
        ).to_labeled_examples()
        
    @classmethod
    def from_csv(
        cls,
        path: str,
        *,
        schema: DatasetSchema,
        strict: bool = True,
        cast_to_str: bool = True,
        **read_csv_kwargs,
    ) -> "TaskDataset":
        """
        Load dataset from a CSV file.

        Parameters
        ----------
        path : str
            Local file path or URL.
        schema : DatasetSchema
            Dataset schema definition.
        strict : bool
            Whether to enforce strict type checking.
        cast_to_str : bool
            Whether to cast required columns to string.
        read_csv_kwargs :
            Forwarded to pandas.read_csv.
        """
        df = pd.read_csv(path, **read_csv_kwargs)

        if cast_to_str:
            for col in schema.all_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str)

        return cls(
            data=df,
            schema=schema,
            strict=strict,
        ).to_labeled_examples()

    @classmethod
    def from_json(
        cls,
        path: str,
        *,
        schema: DatasetSchema,
        strict: bool = True,
        cast_to_str: bool = True,
        orient: Optional[str] = None,
        record_path: Optional[str] = None,
        meta: Optional[list[str]] = None,
        **read_json_kwargs,
    ) -> "TaskDataset":
        """
        Load dataset from a JSON or JSONL file.

        Parameters
        ----------
        path : str
            Local file path or URL.
        schema : DatasetSchema
            Dataset schema definition.
        strict : bool
            Whether to enforce strict type checking.
        cast_to_str : bool
            Whether to cast required columns to string.
        orient : str, optional
            Passed to pandas.read_json.
        record_path : list[str], optional
            Path to nested records for normalization.
        meta : list[str], optional
            Fields to use as metadata when normalizing.
        read_json_kwargs :
            Forwarded to pandas.read_json.
        """

        if record_path is not None:
            df = pd.json_normalize(
                pd.read_json(path, **read_json_kwargs),
                record_path=record_path,
                meta=meta,
            )
        else:
            df = pd.read_json(path, orient=orient, **read_json_kwargs)

        if cast_to_str:
            for col in schema.all_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str)

        return cls(
            data=df,
            schema=schema,
            strict=strict,
        ).to_labeled_examples()


    def _validate(self) -> None:
        missing = set(self.schema.all_columns) - set(self.data.columns)
        if missing:
            raise ValueError(
                f"Dataset is missing required columns: {sorted(missing)}"
            )

        if self.strict:
            self._validate_types()

    def _validate_types(self) -> None:
        for col in self.schema.all_columns:
            if not self.data[col].map(lambda x: isinstance(x, str)).all():
                raise TypeError(
                    f"Column `{col}` must contain only strings."
                )

    @staticmethod
    def _assert_hf_schema(hf_ds, schema: DatasetSchema) -> None:
        missing = set(schema.all_columns) - set(hf_ds.column_names)
        if missing:
            raise ValueError(
                "HuggingFace dataset does not match required schema.\n"
                f"Missing columns: {sorted(missing)}\n"
                f"Available columns: {hf_ds.column_names}"
            )
            
    def to_labeled_examples(
        self,
        *,
        question_id_column: str | None = None
    ) -> Tuple[List[Dict], List[str], List[str]]:
        """
        Convert dataset into flattened (prompt, label) examples.

        Returns
        -------
        examples : List[Dict]
            Flat list of labeled examples.
        eval_prompts : List[str]
            List of original prompts.
        labels : List[str]
            Unique response category labels.
        """

        examples: List[Dict] = []
        eval_prompts: List[str] = []
        labels = list(self.schema.category_columns)

        for idx, row in self.data.iterrows():
            q_text = row[self.schema.prompt_column]

            # Question ID
            if question_id_column is not None:
                q_id = row[question_id_column]
            else:
                q_id = idx

            eval_prompts.append(q_text)

            for lbl in labels:
                response = row[lbl]

                text_parts = []
                text_parts.append(q_text)
                text_parts.append(response)

                examples.append(
                    {
                        "id": f"{q_id}_{lbl}",
                        "original_question": q_text,
                        "text": "\n".join(text_parts),
                        "label": lbl,
                    }
                )

        return examples, eval_prompts



    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]

        return {
            "prompt": row[self.schema.prompt_column],
            "responses": {
                col: row[col]
                for col in self.schema.category_columns
            },
        }


if __name__ == "__main__":
    
    schema = DatasetSchema(
                        prompt_column="Question",
                        category_columns=["Correct Answers", "Incorrect Answers"],
                    )
    
    dataset, eval_prompts, unique_labels = TaskDataset.from_huggingface(repo_id="domenicrosati/TruthfulQA", split="train", schema=schema)
    