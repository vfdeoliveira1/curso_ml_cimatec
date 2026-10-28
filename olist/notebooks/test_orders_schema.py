"""
Testes unitários para o schema pandera de olist_orders_dataset.

Rodar com:
    pytest test_orders_schema.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import pytest

# garante que o import funcione independente de onde o pytest for chamado
sys.path.insert(0, str(Path(__file__).resolve().parent))

from projeto import orders_schema


def _base_df() -> pd.DataFrame:
    """Um DataFrame válido, seguindo exatamente o header do CSV."""
    return pd.DataFrame(
        {
            "order_id": ["ord_001", "ord_002", "ord_003"],
            "customer_id": ["cus_001", "cus_002", "cus_003"],
            "order_status": ["delivered", "shipped", "canceled"],
            "order_purchase_timestamp": pd.to_datetime(
                ["2017-10-02 10:56:33", "2017-10-05 09:00:00", "2017-10-06 12:00:00"]
            ),
            "order_approved_at": pd.to_datetime(
                ["2017-10-02 11:07:15", "2017-10-05 10:00:00", pd.NaT]
            ),
            "order_delivered_carrier_date": pd.to_datetime(
                ["2017-10-04 19:55:00", pd.NaT, pd.NaT]
            ),
            "order_delivered_customer_date": pd.to_datetime(
                ["2017-10-10 21:25:13", pd.NaT, pd.NaT]
            ),
            "order_estimated_delivery_date": pd.to_datetime(
                ["2017-10-18", "2017-10-20", "2017-10-25"]
            ),
        }
    )


def test_schema_aceita_dados_validos():
    df = _base_df()
    validado = orders_schema.validate(df)
    assert len(validado) == 3


def test_schema_rejeita_order_status_invalido():
    df = _base_df()
    df.loc[0, "order_status"] = "status_que_nao_existe"
    with pytest.raises(pa.errors.SchemaError):
        orders_schema.validate(df)


def test_schema_rejeita_order_id_duplicado():
    df = _base_df()
    df.loc[1, "order_id"] = df.loc[0, "order_id"]  # duplica o id
    with pytest.raises(pa.errors.SchemaError):
        orders_schema.validate(df)


def test_schema_rejeita_order_id_nulo():
    df = _base_df()
    df.loc[0, "order_id"] = None
    with pytest.raises(pa.errors.SchemaError):
        orders_schema.validate(df)


def test_schema_rejeita_data_aprovacao_antes_da_compra():
    df = _base_df()
    # aprovação um dia ANTES da compra -> viola a regra de negócio
    df.loc[0, "order_approved_at"] = df.loc[0, "order_purchase_timestamp"] - pd.Timedelta(days=1)
    with pytest.raises(pa.errors.SchemaError):
        orders_schema.validate(df)


def test_schema_rejeita_estimativa_de_entrega_nula():
    df = _base_df()
    df.loc[0, "order_estimated_delivery_date"] = pd.NaT
    with pytest.raises(pa.errors.SchemaError):
        orders_schema.validate(df)