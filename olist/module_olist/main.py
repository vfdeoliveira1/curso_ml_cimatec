import pandas as pd
from pathlib import Path
from loguru import logger

def load_data(orders_path: Path, items_path: Path, customers_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carrega os dados de pedidos, itens e clientes a partir dos arquivos CSV fornecidos.
    """
    orders = pd.read_csv(orders_path, parse_dates=['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date'])
    items = pd.read_csv(items_path)
    customers = pd.read_csv(customers_path)
    
    return orders, items, customers

def save_dataset(dataset: pd.DataFrame, output_path: Path) -> None:
    dataset.to_csv(output_path, index=False)
    logger.success(f"Dataset salvo em: {output_path}")

def create_target(orders: pd.DataFrame) -> pd.DataFrame:
    delivered_orders = orders.loc[
        orders["order_status"].eq("delivered")
        & orders["order_delivered_customer_date"].notna()
        & orders["order_estimated_delivery_date"].notna()
        & orders["order_approved_at"].notna()
    ].copy() 

    delivered_orders["is_late"] = (
        delivered_orders["order_delivered_customer_date"]
        > delivered_orders["order_estimated_delivery_date"]
    ).astype("int8") 

    # CORREÇÃO AQUI: usando logger.info em vez de apenas logger()
    logger.info(f"Pedidos originais: {len(orders):,}")
    logger.info(f"Pedidos no recorte histórico: {len(delivered_orders):,}")
    logger.info("\n" + str(delivered_orders["is_late"].value_counts(dropna=False)))
    
    return delivered_orders

def aggregate_items(items: pd.DataFrame) -> pd.DataFrame:
    items_agg = (
        items.groupby("order_id", as_index=False)
        .agg(
            item_count=("order_item_id", "count"),
            seller_count=("seller_id", "nunique"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
        )
    )

    assert items_agg["order_id"].is_unique
    return items_agg

def create_dataset(orders, items, customers) -> pd.DataFrame:
    orders = create_target(orders)
    items_agg = aggregate_items(items)

    data = orders.merge(
        items_agg, on="order_id", how="left", validate='1:1'
    )
    
    data = data.merge(
        customers[["customer_id", "customer_city"]], on="customer_id", how="left", validate='many_to_many'
    )

    return data

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy() 
    
    data["promised_days"] = (
        data["order_estimated_delivery_date"] 
        - data["order_approved_at"] 
    ).dt.total_seconds().div(86_400) 

    data["purchase_month"] = data["order_purchase_timestamp"].dt.month
    data["purchase_weekday"] = data["order_purchase_timestamp"].dt.dayofweek
    data["purchase_hour"] = data["order_purchase_timestamp"].dt.hour

    return data

def main():
    # 1. Definir os caminhos de entrada (ajuste os nomes/pastas conforme sua estrutura real)
    raw_path = Path("olist/data/raw")
    
    orders_path = raw_path / "olist_orders_dataset.csv"
    items_path = raw_path / "olist_order_items_dataset.csv"
    customers_path = raw_path / "olist_customers_dataset.csv"
    
    # 2. Definir o caminho de saída (pasta interim)
    interim_path = Path("data/interim")
    interim_path.mkdir(parents=True, exist_ok=True)
    output_file = interim_path / "dataset_pre_processado.csv"
    
    try:
        logger.info("Iniciando o carregamento dos dados...")
        orders, items, customers = load_data(orders_path, items_path, customers_path)
        
        logger.info("Realizando as junções e criando a variável-alvo...")
        dataset = create_dataset(orders, items, customers)
        
        logger.info("Criando as features baseadas em tempo...")
        dataset_final = create_features(dataset)
        
        logger.info("Salvando o dataset final...")
        save_dataset(dataset_final, output_file)
        
    except FileNotFoundError as e:
        logger.error(f"Arquivo não encontrado. Verifique se a pasta data/raw e os arquivos existem: {e}")
    except Exception as e:
        logger.error(f"Ocorreu um erro inesperado durante o processamento: {e}")

if __name__ == "__main__":
    main()