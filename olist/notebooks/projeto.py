"""
Teste/validação de schema com pandera para o olist_orders_dataset.csv

Colunas esperadas (header do CSV):
"order_id","customer_id","order_status","order_purchase_timestamp",
"order_approved_at","order_delivered_carrier_date",
"order_delivered_customer_date","order_estimated_delivery_date"

Uso:
    python validar_orders_pandera.py
"""

from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

# --------------------------------------------------------------------------
# 1. Schema pandera para o dataset de pedidos (olist_orders_dataset)
# --------------------------------------------------------------------------

STATUS_VALIDOS = [
    "created",
    "approved",
    "processing",
    "invoiced",
    "shipped",
    "delivered",
    "unavailable",
    "canceled",
]

orders_schema = DataFrameSchema(
    {
        "order_id": Column(str, Check.str_length(min_value=1), unique=True, nullable=False),
        "customer_id": Column(str, Check.str_length(min_value=1), nullable=False),
        "order_status": Column(str, Check.isin(STATUS_VALIDOS), nullable=False),
        "order_purchase_timestamp": Column("datetime64[ns]", nullable=False),
        # datas de aprovação/entrega podem ser nulas (pedido ainda não avançou no fluxo)
        "order_approved_at": Column("datetime64[ns]", nullable=True),
        "order_delivered_carrier_date": Column("datetime64[ns]", nullable=True),
        "order_delivered_customer_date": Column("datetime64[ns]", nullable=True),
        "order_estimated_delivery_date": Column("datetime64[ns]", nullable=False),
    },
    checks=[
        # a data de aprovação (quando existe) não pode ser anterior à compra
        Check(
            lambda df: (df["order_approved_at"].isna())
            | (df["order_approved_at"] >= df["order_purchase_timestamp"]),
            error="order_approved_at anterior a order_purchase_timestamp",
        ),
        # a entrega ao cliente (quando existe) não pode ser anterior à compra
        Check(
            lambda df: (df["order_delivered_customer_date"].isna())
            | (df["order_delivered_customer_date"] >= df["order_purchase_timestamp"]),
            error="order_delivered_customer_date anterior a order_purchase_timestamp",
        ),
    ],
    strict=False,  # troque para True se quiser barrar colunas extras
    coerce=True,
)


# --------------------------------------------------------------------------
# 2. Função de validação reutilizável
# --------------------------------------------------------------------------

DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def carregar_e_validar(caminho_csv: str) -> pd.DataFrame:
    df = pd.read_csv(caminho_csv, parse_dates=DATE_COLS)

    try:
        df_validado = orders_schema.validate(df, lazy=True)
        print(f"✅ Schema válido! {len(df_validado)} linhas conferem com as regras.")
        return df_validado
    except pa.errors.SchemaErrors as exc:
        print("❌ Falhas de validação encontradas:")
        # exc.failure_cases traz um resumo tabular de cada falha
        print(exc.failure_cases.to_string(index=False))
        raise


# --------------------------------------------------------------------------
# 3. Fluxo original do usuário, agora com validação antes do profiling
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from data_profiling import ProfileReport 

    caminho_entrada = "olist/data/raw/olist_orders_dataset.csv"
    caminho_saida = Path("olist/data/processed/relatorio.html")
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    df = carregar_e_validar(caminho_entrada)

    profile = ProfileReport(df, title="Relatório de Perfil de Dados")
    profile.to_file(str(caminho_saida))