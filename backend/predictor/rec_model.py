import os
import sys
import copy
import json
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import defaultdict, Counter

from typing import Dict, List, Any

from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn

from torch import Tensor

from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class BaseModel:

    def __init__(self) -> None:
        self._item_id_feature_name = 'item_id'
        self._receipt_id_feature_name = 'receipt_id'

        self._num_items = None
        self._item_2_id = None
        self._id_2_item = None

    def fit(self, df: pd.DataFrame) -> None:
        df = self._preprocess_dataset(df)

        self._num_items = 0
        self._item_id_2_new_id = {}
        self._new_id_2_item_id = {}

        for _, row in df.iterrows():
            item_id = row[self._item_id_feature_name]
            if item_id not in self._item_id_2_new_id:
                self._item_id_2_new_id[item_id] = self._num_items
                self._new_id_2_item_id[self._num_items] = item_id
                self._num_items += 1

        return self

    def predict_sample(self, sample: List[Any]) -> int:
        raise NotImplemetedError

    def predict_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._preprocess_dataset(df)
        samples = self._dataset_to_samples(df)

        result = defaultdict(list)
        visited = set()
        for receipt_id, sample in samples.items():
            item_id = self.predict_sample(sample)

            assert receipt_id not in visited
            visited.add(receipt_id)
            result[self._receipt_id_feature_name].append(receipt_id)
            result[self._item_id_feature_name].append(self.predict_sample(sample))

        return pd.DataFrame(result)

    def _preprocess_sample(self, sample: List[Any]) -> List[Any]:
        new_sample = copy.deepcopy(sample)
        for item in new_sample:
            if item[self._item_id_feature_name] in self._item_id_2_new_id:
                item[self._item_id_feature_name] = self._item_id_2_new_id[item[self._item_id_feature_name]]
            else:
                item[self._item_id_feature_name] = self._num_items

        return new_sample

    def _preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        return copy.deepcopy(df)

    def _dataset_to_samples(self, df: pd.DataFrame) -> Dict[int, List[Any]]:
        samples = defaultdict(list)

        for _, row in df.iterrows():
            receipt_id = row.pop(self._receipt_id_feature_name)
            samples[receipt_id].append(dict(row))

        return samples


class PopModel(BaseModel):

    def __init__(self) -> None:
        super().__init__()
        self._popular_items = None

    def fit(self, df: pd.DataFrame) -> BaseModel:
        super().fit(df)

        df = self._preprocess_dataset(df)
        samples = self._dataset_to_samples(df)

        item_id_counts = Counter()
        for sample in samples.values():
            for item in self._preprocess_sample(sample):
                item_id_counts[item[self._item_id_feature_name]] += 1

        self._popular_items = sorted(list(item_id_counts.items()), key=lambda x: x[1], reverse=True)

        return self

    def predict_sample(self, sample: List[Any]) -> int:
        if self._num_items is None:
            raise ValueError('Model isn\'t fitted yet!')

        sample = self._preprocess_sample(sample)
        sample_items = set([item[self._item_id_feature_name] for item in sample])

        pred_item_id = None
        for popular_item_id, popular_item_count in self._popular_items:
            if popular_item_id not in sample_items:
                pred_item_id = popular_item_id
                break
        else:
            assert False, 'This is really strange'
        #             pred_item_id = self._popular_items[0][0]

        return self._new_id_2_item_id[pred_item_id]


class TorchModel(BaseModel):

    def __init__(self) -> None:
        super().__init__()
        self._sample_features_to_take = {
            self._item_id_feature_name: torch.long,
            self._price_feature_name: torch.float,
            self._is_present_feature_name: torch.long,
            self._day_feature_name: torch.long,
            self._month_feature_name: torch.long,
            self._hour_feature_name: torch.long,
            self._quantity_feature_name: torch.float
        }
        self._aggregated_features_to_take = {
            self._sum_price_feature_name: torch.float,
            self._device_id_feature_name: torch.long
        }
        self._target_features_to_take = {
            self._item_id_feature_name: torch.long
        }

    def _create_training_dataset(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        augmented_samples = []

        for _, sample in samples.items():
            sample = self._preprocess_sample(sample)
            extented_sample = sample * 2
            for i in range(len(sample)):
                augmented_sample = extented_sample[i: i + len(sample) - 1]
                assert len(augmented_sample) == len(sample) - 1
                target_item = extented_sample[i + len(sample) - 1]

                if len(augmented_sample) > 0:
                    converted_sample = self._convert_sample(augmented_sample)

                    # Target features
                    for feature_to_take in self._target_features_to_take:
                        values_to_take = [x[feature_to_take] for x in [target_item]]
                        converted_sample.update({
                            f'target_{feature_to_take}': values_to_take,
                            f'target_{feature_to_take}.length': len(values_to_take)
                        })

                    augmented_samples.append(converted_sample)

        return augmented_samples

    def _convert_sample(self, sample: List[Dict[str, Any]]) -> Dict[str, Any]:
        converted_sample = {}

        # Convert item-level features
        for feature_to_take in self._sample_features_to_take:
            values_to_take = [x[feature_to_take] for x in sample]
            converted_sample.update({
                feature_to_take: values_to_take,
                f'{feature_to_take}.length': len(values_to_take)
            })

        # Convert sample-level features
        # sum price
        converted_sample.update({
            self._sum_price_feature_name: [sum([x for x in converted_sample[self._price_feature_name]])],
            f'{self._sum_price_feature_name}.length': 1
        })
        # device id
        converted_sample.update({
            self._device_id_feature_name: [sample[0][self._device_id_feature_name]],
            f'{self._device_id_feature_name}.length': 1
        })

        return converted_sample

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        processed_batch = {}

        for key in batch[0].keys():
            if not key.endswith('.length'):
                assert '{}.length'.format(key) in batch[0]

                processed_batch[key] = []
                processed_batch[f'{key}.length'] = []

                for sample in batch:
                    processed_batch[key].extend(sample[key])
                    processed_batch[f'{key}.length'].append(sample[f'{key}.length'])

        for part, values in processed_batch.items():
            if part.endswith('.length'):
                processed_batch[part] = torch.tensor(values, dtype=torch.long)
            elif part in set(self._sample_features_to_take):
                processed_batch[part] = torch.tensor(values, dtype=self._sample_features_to_take[part])
            elif part in set(self._aggregated_features_to_take):
                processed_batch[part] = torch.tensor(values, dtype=self._aggregated_features_to_take[part])
            elif part[7:] in set(self._target_features_to_take):
                processed_batch[part] = torch.tensor(values, dtype=self._target_features_to_take[part[7:]])
            else:
                assert False

        return processed_batch


class RandomModel(BaseModel):

    def predict_sample(self, sample: List[Any]) -> int:
        if self._num_items is None:
            raise ValueError('Model isn\'t fitted yet!')

        sample = self._preprocess_sample(sample)
        sample_items = set([item[self._item_id_feature_name] for item in sample])

        item_id = np.random.randint(0, self._num_items)
        while item_id in sample_items:  # TODO[vbaikalov] ask about it
            item_id = np.random.randint(0, self._num_items)

        return self._new_id_2_item_id[item_id]


class TransferModel(BaseModel):

    def __init__(self) -> None:
        super().__init__()
        self._item_transfers = None

    def fit(self, df: pd.DataFrame) -> BaseModel:
        super().fit(df)
        df = self._preprocess_dataset(df)
        samples = self._dataset_to_samples(df)

        self._item_transfers = defaultdict(Counter)
        for sample in samples.values():
            preprocessed_sample = self._preprocess_sample(sample)
            item_ids = set([x[self._item_id_feature_name] for x in preprocessed_sample])

            for fst_item_id in item_ids:
                for snd_item_id in item_ids:
                    if fst_item_id != snd_item_id:
                        self._item_transfers[fst_item_id][snd_item_id] += 1
                        self._item_transfers[snd_item_id][fst_item_id] += 1
        return self

    def predict_sample(self, sample: List[Any]) -> int:
        if self._num_items is None:
            raise ValueError('Model isn\'t fitted yet!')

        sample = self._preprocess_sample(sample)
        sample_item_ids = set([item[self._item_id_feature_name] for item in sample])

        adjacened_samples = Counter()
        for item_id in sample_item_ids:
            for adjacened_item_id, cnt in self._item_transfers[item_id].items():
                adjacened_samples[adjacened_item_id] += cnt

        popular_samples = sorted(adjacened_samples.items(), key=lambda x: x[1], reverse=True)
        for popular_item_id, popular_item_count in popular_samples:
            if popular_item_id not in sample_item_ids:
                pred_item_id = popular_item_id
                break
        else:
            pred_item_id = np.random.randint(0, self._num_items)
            while pred_item_id in sample_item_ids:
                pred_item_id = np.random.randint(0, self._num_items)

        return self._new_id_2_item_id[pred_item_id]


class TorchEncoder(nn.Module):

    @torch.no_grad()
    def _init_weights(self, initializer_range) -> None:
        for key, value in self.named_parameters():
            if 'weight' in key:
                if 'norm' in key:
                    nn.init.ones_(value.data)
                else:
                    nn.init.trunc_normal_(
                        value.data,
                        std=initializer_range,
                        a=-2 * initializer_range,
                        b=2 * initializer_range
                    )
            elif 'bias' in key:
                nn.init.zeros_(value.data)
            else:
                raise ValueError(f'Unknown transformer weight: {key}')

    @staticmethod
    def _create_masked_tensor(data, lengths):
        batch_size = lengths.shape[0]
        max_sequence_length = lengths.max().item()

        padded_embeddings = torch.zeros(
            batch_size, max_sequence_length, data.shape[-1],
            dtype=torch.float, device=DEVICE
        )  # [batch_size, max_seq_len, emb_dim]

        mask = torch.arange(
            end=max_sequence_length,
            device=DEVICE
        )[None].tile([batch_size, 1]) < lengths[:, None]  # (batch_size, max_seq_len)

        padded_embeddings[mask] = data

        return padded_embeddings, mask


class NNModel(TorchModel):

    def __init__(
            self,
            num_epochs: int,
            batch_size: int = 256,
            learning_rate: float = 3e-4,
    ) -> None:
        super().__init__()
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._encoder = None

    def predict_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._preprocess_dataset(df)
        samples = self._dataset_to_samples(df)

        processed_samples = []
        processed_samples_receipt_ids = []
        for receipt_id, sample in samples.items():
            processed_samples_receipt_ids.append(receipt_id)
            processed_samples.append(self._convert_sample(self._preprocess_sample(sample)))

        result = defaultdict(list)
        for idx in range(0, len(processed_samples), self._batch_size):
            batch_recept_ids = processed_samples_receipt_ids[idx: idx + self._batch_size]
            batch = self._collate_fn(processed_samples[idx: idx + self._batch_size])
            for key, values in batch.items():
                batch[key] = batch[key].to(DEVICE)

            with torch.no_grad():
                items_logits = self._encoder(batch)  # [batch_size, num_items]
                items_logits[:, self._num_items] = -torch.inf
                pred_item_ids = torch.topk(items_logits, k=self._num_items - 1,
                                           dim=-1).indices.cpu().tolist()  # [batch_size, 100]

                pred_items = []
                for processed_sample, top_n in zip(processed_samples[idx: idx + self._batch_size], pred_item_ids):
                    sample_items = set(processed_sample[self._item_id_feature_name])

                    for popular_item_id in top_n:
                        if popular_item_id not in sample_items:
                            pred_items.append(popular_item_id)
                            break
                    else:
                        assert False, 'This is really strange'

                pred_items = [self._new_id_2_item_id[pred_item] for pred_item in pred_items]

            result[self._receipt_id_feature_name].extend(batch_recept_ids)
            result[self._item_id_feature_name].extend(pred_items)

        return pd.DataFrame(result)

    def fit(self, df: pd.DataFrame) -> BaseModel:
        super().fit(df)

        df = self._preprocess_dataset(df)
        samples = self._dataset_to_samples(df)

        train_samples = {}
        test_samples = {}
        for sample_id, sample in samples.items():
            p = np.random.uniform(low=0.0, high=1.0)
            if p < 0.7:
                train_samples[sample_id] = sample
            else:
                test_samples[sample_id] = sample

        train_dataset = self._create_training_dataset(train_samples)
        test_dataset = self._create_training_dataset(test_samples)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self._collate_fn
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self._collate_fn
        )

        self._encoder = self._init_encoder().to(DEVICE)

        optimizer = torch.optim.AdamW(self._encoder.parameters(), lr=self._learning_rate)
        loss_function = torch.nn.CrossEntropyLoss()

        step_num = 0
        epoch_num = 0
        while epoch_num < self._num_epochs:
            print(epoch_num)
            for batch in train_dataloader:
                self._encoder.train()

                batch_ = copy.deepcopy(batch)
                for key, values in batch_.items():
                    batch_[key] = batch_[key].to(DEVICE)

                next_item_logits = self._encoder(batch_)  # [batch_size, num_items]
                loss = loss_function(next_item_logits, batch_['target_item_id'])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step_num += 1

                if step_num % 100 == 0:
                    with torch.no_grad():
                        accuracies = []
                        losses = []

                        for batch in test_dataloader:
                            self._encoder.eval()

                            batch_ = copy.deepcopy(batch)
                            for key, values in batch_.items():
                                batch_[key] = batch_[key].to(DEVICE)

                            next_item_logits = self._encoder(batch_)  # [batch_size, num_items]
                            loss = loss_function(next_item_logits, batch_['target_item_id'])

                            pred_item_ids = torch.argmax(next_item_logits, dim=-1).cpu().tolist()  # [batch_size]

                            losses.append(loss.item())
                            accuracies.append(accuracy_score(batch_['target_item_id'].cpu().tolist(), pred_item_ids))

                    print(np.mean(losses))
                    print(np.mean(accuracies))
                    print()

            epoch_num += 1

        return self

    def predict_sample(self, sample: List[Any]) -> int:
        if self._encoder is None:
            raise ValueError('Model isn\'t fitted yet!')
        self._encoder.eval()

        sample = self._preprocess_sample(sample)

        # TODO[vbaikalov] re-write
        samples = []
        for feature_to_take in self._features_to_take:
            values_to_take = [x[feature_to_take] for x in sample]
            samples.append({
                feature_to_take: values_to_take,
                f'{feature_to_take}.length': len(values_to_take)
            })
        batch = self._collate_fn(samples)
        for key, values in batch.items():
            batch[key] = batch[key].to(DEVICE)

        with torch.no_grad():
            items_logits = self._encoder(batch)  # [1, num_items]
            items_logits[:, self._num_items] = -torch.inf
            pred_item_id = torch.argmax(items_logits, dim=-1).squeeze().cpu().item()  # TODO[me] fix

        return self._new_id_2_item_id[pred_item_id]

    def _init_encoder(self) -> TorchEncoder:
        raise NotImplementedError


class MatrixFactorizationEncoder(TorchEncoder):

    def __init__(
            self,
            item_idx_prefix: str,
            price_prefix: str,
            device_idx_prefix: str,
            hour_prefix: str,
            day_prefix: str,
            month_prefix: str,
            quantity_prefix: str,
            sum_price_prefix: str,
            is_present_prefix: str,
            num_items: int,
            num_devices: int,
            embedding_dim: int = 256,
            dropout: float = 0.0,
            layer_norm_eps: float = 1e-6,
            initializer_range: float = 0.02
    ) -> None:
        super().__init__()
        self._item_idx_prefix = item_idx_prefix
        self._price_prefix = price_prefix
        self._device_idx_prefix = device_idx_prefix
        self._hour_prefix = hour_prefix
        self._day_prefix = day_prefix
        self._month_prefix = month_prefix
        self._quantity_prefix = quantity_prefix
        self._sum_price_prefix = sum_price_prefix
        self._is_present_prefix = is_present_prefix

        self._num_items = num_items
        self._embedding_dim = embedding_dim
        self._dropout = dropout
        self._initializer_range = initializer_range

        self._item_embeddings = nn.Embedding(
            num_embeddings=num_items + 1,
            embedding_dim=embedding_dim
        )  # +1 for unknown items

        self._device_embeddings = nn.Embedding(
            num_embeddings=num_devices + 1,
            embedding_dim=embedding_dim
        )

        self._price_projection = nn.Linear(
            in_features=1,
            out_features=embedding_dim
        )

        self._sum_price_projection = nn.Linear(
            in_features=1,
            out_features=embedding_dim
        )

        self._is_present_projection = nn.Embedding(
            num_embeddings=2,
            embedding_dim=embedding_dim
        )

        self._hour_projection = nn.Embedding(
            num_embeddings=25,
            embedding_dim=embedding_dim
        )

        self._day_projection = nn.Embedding(
            num_embeddings=32,
            embedding_dim=embedding_dim
        )

        self._month_projection = nn.Embedding(
            num_embeddings=13,
            embedding_dim=embedding_dim
        )

        self._quantity_projection = nn.Linear(
            in_features=1,
            out_features=embedding_dim
        )

        self._layernorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self._dropout = nn.Dropout(dropout)

        self._head = nn.Linear(in_features=embedding_dim, out_features=num_items + 1)

        self._init_weights(initializer_range)

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        item_ids = inputs[self._item_idx_prefix]  # [all_items]
        item_lengths = inputs[f'{self._item_idx_prefix}.length']  # [batch_size]
        item_embeddings = self._item_embeddings(item_ids)  # [all_items, embedding_dim]

        if self._price_prefix is not None:
            price = inputs[self._price_prefix]  # [all_items]
            price_embeddings = self._price_projection(price.unsqueeze(dim=-1))  # [all_items, embedding_dim]
            item_embeddings += price_embeddings

        if self._is_present_prefix is not None:
            is_present = inputs[self._is_present_prefix]  # [all_items]
            is_present_embeddings = self._is_present_projection(is_present)  # [all_items, embedding_dim]
            item_embeddings += is_present_embeddings

        if self._hour_prefix is not None:
            hour = inputs[self._hour_prefix]  # [all_items]
            hour_embeddings = self._hour_projection(hour)  # [all_items, embedding_dim]
            item_embeddings += hour_embeddings

        if self._day_prefix is not None:
            day = inputs[self._day_prefix]  # [all_items]
            day_embeddings = self._day_projection(day)  # [all_items, embedding_dim]
            item_embeddings += day_embeddings

        if self._month_prefix is not None:
            month = inputs[self._month_prefix]  # [all_items]
            month_embeddings = self._month_projection(month)  # [all_items, embedding_dim]
            item_embeddings += month_embeddings

        if self._quantity_prefix is not None:
            quantity = inputs[self._quantity_prefix]  # [all_items]
            quantity_embeddings = self._quantity_projection(
                quantity_embeddings.unsqueeze(dim=-1))  # [all_items, embedding_dim]
            item_embeddings += quantity_embeddings

        embeddings, mask = self._create_masked_tensor(
            data=item_embeddings,
            lengths=item_lengths
        )  # [batch_size, seq_len, embedding_dim], [batch_size, seq_len]

        embeddings = self._layernorm(embeddings)  # (batch_size, seq_len, embedding_dim)
        embeddings = self._dropout(embeddings)  # (batch_size, seq_len, embedding_dim)

        embeddings[~mask] = 0

        check_embedding = torch.mean(embeddings, dim=1)  # (batch_size, embedding_dim)
        if self._sum_price_prefix is not None:
            sum_price = inputs[self._sum_price_prefix]  # [batch_size]
            sum_price_embedding = self._sum_price_projection(sum_price.unsqueeze(dim=-1))  # [batch_size, embedding_dim]
            check_embedding += sum_price_embedding  # [batch_size, embedding_dim]
        if self._device_idx_prefix is not None:
            device_id = inputs[self._device_idx_prefix]  # [batch_size]
            device_id_embedding = self._device_embeddings(device_id)  # [batch_size, embedding_dim]
            check_embedding += device_id_embedding

        items_logits = self._head(check_embedding)  # (batch_size, num_items)

        return items_logits


class MatrixFactorizationModel(NNModel):

    def __init__(
            self,
            num_epochs: int,
            batch_size: int = 256,
            learning_rate: float = 3e-4,
            embedding_dim: int = 256,
            dropout: float = 0.0,
            initializer_range: float = 0.02,
            use_price_feature: bool = False,
            use_sum_price_feature: bool = False,
            use_device_id: bool = False,
            use_is_present_feature: bool = False,
            use_hour_feature: bool = False,
            use_day_feature: bool = False,
            use_month_feature: bool = False,
            use_quantity_feature: bool = False
    ) -> None:
        super().__init__(
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        self._embedding_dim = embedding_dim
        self._dropout = dropout
        self._initializer_range = initializer_range

        self._use_price_feature = use_price_feature
        self._use_sum_price_feature = use_sum_price_feature
        self._use_device_id = use_device_id
        self._use_is_present_feature = use_is_present_feature
        self._use_hour_feature = use_hour_feature
        self._use_day_feature = use_day_feature
        self._use_month_feature = use_month_feature
        self._use_quantity_feature = use_quantity_feature

    def _init_encoder(self) -> TorchEncoder:
        return MatrixFactorizationEncoder(
            item_idx_prefix=self._item_id_feature_name,
            price_prefix=self._price_feature_name if self._use_price_feature else None,
            sum_price_prefix=self._sum_price_feature_name if self._use_sum_price_feature else None,
            device_idx_prefix=self._device_id_feature_name if self._use_device_id else None,
            is_present_prefix=self._is_present_feature_name if self._use_is_present_feature else None,
            hour_prefix=self._hour_feature_name if self._use_hour_feature else None,
            day_prefix=self._day_feature_name if self._use_day_feature else None,
            month_prefix=self._month_feature_name if self._use_month_feature else None,
            quantity_prefix=self._quantity_feature_name if self._use_quantity_feature else None,
            num_items=self._num_items,
            num_devices=self._num_devices,
            embedding_dim=self._embedding_dim,
            dropout=self._dropout,
            initializer_range=self._initializer_range
        )


class AttentionEncoder(TorchEncoder):
    def __init__(
            self,
            item_idx_prefix: str,
            price_prefix: str,
            num_items: int,
            num_layers: int = 2,
            num_heads: int = 2,
            embedding_dim: int = 256,
            dropout: float = 0.0,
            layer_norm_eps: float = 1e-6,
            initializer_range: float = 0.02
    ) -> None:
        super().__init__()
        self._item_idx_prefix = item_idx_prefix
        self._price_prefix = price_prefix
        self._num_items = num_items
        self._num_layers = num_layers
        self._embedding_dim = embedding_dim
        self._dropout = dropout
        self._layer_norm_eps = layer_norm_eps
        self._initializer_range = initializer_range

        self._item_embeddings = nn.Embedding(
            num_embeddings=num_items + 1,
            embedding_dim=embedding_dim
        )  # +1 for unknown items

        self._price_projection = nn.Linear(
            in_features=1,
            out_features=embedding_dim
        )

        self._layernorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self._dropout = nn.Dropout(dropout)

        self._cls_token = nn.Embedding(
            num_embeddings=1,
            embedding_dim=embedding_dim
        )

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
            activation='relu',
            layer_norm_eps=layer_norm_eps,
            batch_first=True
        )
        self._transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers)

        self._head = nn.Linear(in_features=embedding_dim, out_features=num_items + 1)

        self._init_weights(initializer_range)

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        item_ids = inputs[self._item_idx_prefix]  # [all_items]
        item_lengths = inputs[f'{self._item_idx_prefix}.length']  # [batch_size]
        item_embeddings = self._item_embeddings(item_ids)  # [all_items, embedding_dim]

        if self._price_prefix is not None:
            price = inputs[self._price_prefix]  # [all_items]
            price_embeddings = self._price_projection(price.unsqueeze(dim=-1))  # [all_items, embedding_dim]
            item_embeddings += price_embeddings

        batch_size = item_lengths.shape[0]
        embeddings, mask = self._create_masked_tensor(
            data=item_embeddings,
            lengths=item_lengths
        )  # [batch_size, seq_len, embedding_dim], [batch_size, seq_len]

        embeddings = self._layernorm(embeddings)  # (batch_size, seq_len, embedding_dim)
        embeddings = self._dropout(embeddings)  # (batch_size, seq_len, embedding_dim)

        embeddings[~mask] = 0

        cls_token = torch.tile(
            self._cls_token.weight[None, ...],  # (1, 1, embedding_dim)
            dims=[batch_size, 1, 1]
        )  # (batch_size, 1, embedding_dim)
        embeddings = torch.cat([cls_token, embeddings], dim=1)
        mask = torch.cat([torch.ones(batch_size, 1).to(mask.device), mask], dim=1).bool()

        embeddings = self._transformer_encoder(
            src=embeddings,
            src_key_padding_mask=~mask
        )  # (batch_size, seq_len, embedding_dim)

        check_embedding = embeddings[:, 0, :]  # (batch_size, embedding_dim)
        items_logits = self._head(check_embedding)  # (batch_size, num_items)

        return items_logits


class AttentionModel(NNModel):

    def __init__(
            self,
            num_epochs: int = 1,
            batch_size: int = 256,
            learning_rate: float = 3e-4,
            num_layers: int = 2,
            num_heads: int = 2,
            embedding_dim: int = 256,
            dropout: float = 0.0,
            layer_norm_eps: float = 1e-6,
            initializer_range: float = 0.02,
            use_price_feature: bool = False
    ) -> None:
        super().__init__(
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._embedding_dim = embedding_dim
        self._dropout = dropout
        self._layer_norm_eps = layer_norm_eps
        self._initializer_range = initializer_range
        self._use_price_feature = use_price_feature

    def _init_encoder(self) -> TorchEncoder:
        return AttentionEncoder(
            item_idx_prefix=self._item_id_feature_name,
            price_prefix=self._price_feature_name if self._use_price_feature else None,
            num_items=self._num_items,
            num_layers=self._num_layers,
            num_heads=self._num_heads,
            embedding_dim=self._embedding_dim,
            dropout=self._dropout,
            initializer_range=self._initializer_range
        )


def labels_to_dict(df_labels: pd.DataFrame) -> Dict[int, int]:
    result = {}
    for _, row in df_labels.iterrows():
        receipt_id = row['receipt_id']
        item_id = row['item_id']
        if receipt_id not in result:
            result[receipt_id] = item_id
        else:
            if item_id != result[receipt_id]:
                raise ValueError('Repeating receipt_id found!')

    return result


def compute_accuracy(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> float:
    true_results = labels_to_dict(df_true)
    pred_results = labels_to_dict(df_pred)

    if len(true_results) != len(pred_results):
        print('Predictions size mismatch!')

    y_true, y_pred = [], []
    for receipt_id in set(pred_results):
        y_true.append(true_results[receipt_id])
        y_pred.append(pred_results[receipt_id])

    return accuracy_score(y_true, y_pred)
