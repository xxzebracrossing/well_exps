import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional

from torch.utils.data import DataLoader, DistributedSampler

from the_well.data.augmentation import Augmentation
from the_well.data.datasets import WellDataset
from the_well.data.normalization import ZScoreNormalization

logger = logging.getLogger(__name__)


class AbstractDataModule(ABC):
    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def rollout_val_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def rollout_test_dataloader(self) -> DataLoader:
        raise NotImplementedError


class WellDataModule(AbstractDataModule):
    """Data module class to yield batches of samples.

    Args:
        well_base_path:
            Path to the data folder containing the splits (train, validation, and test).
        well_dataset_name:
            Name of the well dataset to use.
        batch_size:
            Size of the batches yielded by the dataloaders
        ---
        include_filters:
            Only file names containing any of these strings will be included.
        exclude_filters:
            File names containing any of these strings will be excluded.
        use_normalization:
            Whether to use normalization on the data.
        normalization_type:
            What kind of normalization to use if use_normalization is True. Currently supports zscore and rms.
        train_dataset:
            What type of training dataset type. WellDataset or DeltaWellDataset options.
        max_rollout_steps:
            Maximum number of steps to use for the rollout dataset. Mostly for memory reasons.
        n_steps_input:
            Number of steps to use as input.
        n_steps_output:
            Number of steps to use as output.
        min_dt_stride:
            Minimum stride in time to use for the dataset.
        max_dt_stride:
            Maximum stride in time to use for the dataset. If this is greater than min, randomly choose between them.
                Note that this is unused for validation/test which uses "min_dt_stride" for both the min and max.
        world_size:
            Number of GPUs in use for distributed training.
        data_workers:
            Number of workers to use for data loading.
        rank:
            Rank of the current process in distributed training.
        transform:
            Augmentation to apply to the data. If None, no augmentation is applied.
        dataset_kws:
            Additional keyword arguments to pass to each dataset, as a dict of dicts.
        storage_kwargs:
            Storage options passed to fsspec for accessing the raw data.
    """

    def __init__(
        self,
        well_base_path: str,
        well_dataset_name: str,
        batch_size: int,
        path: Optional[str] = None,
        include_filters: List[str] = [],
        exclude_filters: List[str] = [],
        use_normalization: bool = False,
        normalization_type: Optional[Callable[..., Any]] = None,
        train_dataset: Callable[..., Any] = WellDataset,
        max_rollout_steps: int = 100,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        min_dt_stride: int = 1,
        max_dt_stride: int = 1,
        world_size: int = 1,
        data_workers: int = 4,
        rank: int = 1,
        boundary_return_type: Literal["padding", None] = "padding",
        transform: Optional[Augmentation] = None,
        dataset_kws: Optional[
            Dict[
                Literal["train", "val", "rollout_val", "test", "rollout_test"],
                Dict[str, Any],
            ]
        ] = None,
        storage_kwargs: Optional[Dict] = None,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("always")  # Ensure warnings are always displayed

            if use_normalization:
                warnings.warn(
                    "`use_normalization` parameter will be removed in a future version. "
                    "For proper normalizing, set both use_normalization=True and normalization_type to either ZScoreNormalization or RMSNormalization."
                    "Default behavior is `normalization_type=ZScoreNormalization` and `use_normalization=True`."
                    "To switch off normalization instead, please set use_normalization=False in the config.yaml file",
                    DeprecationWarning,
                )
                if normalization_type is None:
                    warnings.warn(
                        "use_normalization=True, but normalization_type is None. "
                        "Defaulting to ZScoreNormalization.",
                        UserWarning,
                    )
                    normalization_type = ZScoreNormalization  # Default fallback

            elif normalization_type is not None:
                warnings.warn(
                    "Inconsistent normalization settings: `use_normalization=False`, but `normalization_type` is set. "
                    "Defaulting `normalization_type=None` and `use_normalization=False`.",
                    UserWarning,
                )
                normalization_type = None

        # DeltaWellDataset only for training for delta case, WellDataset for everything else
        self.train_dataset = train_dataset(
            path=path,
            well_base_path=well_base_path,
            well_dataset_name=well_dataset_name,
            well_split_name="train",
            include_filters=include_filters,
            exclude_filters=exclude_filters,
            use_normalization=use_normalization,
            normalization_type=normalization_type,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            storage_options=storage_kwargs,
            min_dt_stride=min_dt_stride,
            max_dt_stride=max_dt_stride,
            boundary_return_type=boundary_return_type,
            transform=transform,
            **(
                dataset_kws["train"]
                if dataset_kws is not None and "train" in dataset_kws
                else {}
            ),
        )
        self.val_dataset = WellDataset(
            path=path,
            well_base_path=well_base_path,
            well_dataset_name=well_dataset_name,
            well_split_name="valid",
            include_filters=include_filters,
            exclude_filters=exclude_filters,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            storage_options=storage_kwargs,
            min_dt_stride=min_dt_stride,
            max_dt_stride=min_dt_stride,
            boundary_return_type=boundary_return_type,
            **(
                dataset_kws["val"]
                if dataset_kws is not None and "val" in dataset_kws
                else {}
            ),
        )
        self.rollout_val_dataset = WellDataset(
            path=path,
            well_base_path=well_base_path,
            well_dataset_name=well_dataset_name,
            well_split_name="valid",
            include_filters=include_filters,
            exclude_filters=exclude_filters,
            max_rollout_steps=max_rollout_steps,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            full_trajectory_mode=True,
            storage_options=storage_kwargs,
            min_dt_stride=min_dt_stride,
            max_dt_stride=min_dt_stride,
            boundary_return_type=boundary_return_type,
            **(
                dataset_kws["rollout_val"]
                if dataset_kws is not None and "rollout_val" in dataset_kws
                else {}
            ),
        )
        self.test_dataset = WellDataset(
            path=path,
            well_base_path=well_base_path,
            well_dataset_name=well_dataset_name,
            well_split_name="test",
            include_filters=include_filters,
            exclude_filters=exclude_filters,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            storage_options=storage_kwargs,
            min_dt_stride=min_dt_stride,
            max_dt_stride=min_dt_stride,
            boundary_return_type=boundary_return_type,
            **(
                dataset_kws["test"]
                if dataset_kws is not None and "test" in dataset_kws
                else {}
            ),
        )
        self.rollout_test_dataset = WellDataset(
            path=path,
            well_base_path=well_base_path,
            well_dataset_name=well_dataset_name,
            well_split_name="test",
            include_filters=include_filters,
            exclude_filters=exclude_filters,
            max_rollout_steps=max_rollout_steps,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            full_trajectory_mode=True,
            storage_options=storage_kwargs,
            min_dt_stride=min_dt_stride,
            max_dt_stride=min_dt_stride,
            boundary_return_type=boundary_return_type,
            **(
                dataset_kws["rollout_test"]
                if dataset_kws is not None and "rollout_test" in dataset_kws
                else {}
            ),
        )
        self.well_base_path = well_base_path
        self.well_dataset_name = well_dataset_name
        self.batch_size = batch_size
        self.world_size = world_size
        self.data_workers = data_workers
        self.rank = rank

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def train_dataloader(self) -> DataLoader:
        """Generate a dataloader for training data.

        Returns:
            A dataloader
        """
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            logger.debug(
                f"Use {sampler.__class__.__name__} "
                f"({self.rank}/{self.world_size}) for training data"
            )
        shuffle = sampler is None

        return DataLoader(
            self.train_dataset,
            num_workers=self.data_workers,
            pin_memory=True,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        """Generate a dataloader for validation data.

        Returns:
            A dataloader
        """
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            logger.debug(
                f"Use {sampler.__class__.__name__} "
                f"({self.rank}/{self.world_size}) for validation data"
            )
        shuffle = sampler is None  # Most valid epochs are short
        return DataLoader(
            self.val_dataset,
            num_workers=self.data_workers,
            pin_memory=True,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            sampler=sampler,
        )

    def rollout_val_dataloader(self) -> DataLoader:
        """Generate a dataloader for rollout validation data.

        Returns:
            A dataloader
        """
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                self.rollout_val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,  # Since we're subsampling, don't want continuous
            )
            logger.debug(
                f"Use {sampler.__class__.__name__} "
                f"({self.rank}/{self.world_size}) for rollout validation data"
            )
        shuffle = sampler is None  # Most valid epochs are short
        return DataLoader(
            self.rollout_val_dataset,
            num_workers=self.data_workers,
            pin_memory=True,
            batch_size=1,
            shuffle=shuffle,  # Shuffling because most batches we take a small subsample
            drop_last=True,
            sampler=sampler,
        )

    def test_dataloader(self) -> DataLoader:
        """Generate a dataloader for test data.

        Returns:
            A dataloader
        """
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                self.test_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
            logger.debug(
                f"Use {sampler.__class__.__name__} "
                f"({self.rank}/{self.world_size}) for test data"
            )
        return DataLoader(
            self.test_dataset,
            num_workers=self.data_workers,
            pin_memory=True,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            sampler=sampler,
        )

    def rollout_test_dataloader(self) -> DataLoader:
        """Generate a dataloader for rollout test data.

        Returns:
            A dataloader
        """
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                self.rollout_test_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
            logger.debug(
                f"Use {sampler.__class__.__name__} "
                f"({self.rank}/{self.world_size}) for rollout test data"
            )
        return DataLoader(
            self.rollout_test_dataset,
            num_workers=self.data_workers,
            pin_memory=True,
            batch_size=1,  # min(self.batch_size, len(self.rollout_test_dataset)),
            shuffle=False,
            drop_last=True,
            sampler=sampler,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.well_dataset_name} on {self.well_base_path}>"
