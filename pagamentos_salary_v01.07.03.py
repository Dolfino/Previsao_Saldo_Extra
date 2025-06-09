import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine

# Configuração do logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_DATA_INICIO = os.getenv("DATA_INICIO", "2024-01-01")
DEFAULT_DATA_FIM = os.getenv("DATA_FIM", "2025-06-08")

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Constantes
RESULTADOS_DIR = os.path.expanduser(
    os.getenv("RESULTADOS_DIR", "~/Documents/Resultados_Scripts_Python")
)


def criar_engine(database: str) -> Optional[Engine]:
    """Cria e retorna uma engine SQLAlchemy para conexão com o banco de dados PostgreSQL.

    Args:
        database: Nome do banco de dados

    Returns:
        Engine configurada ou None em caso de falha

    Raises:
        ValueError: Se variáveis de ambiente estiverem faltando
    """
    try:
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT", "5432")

        if not all([user, password, host]):
            raise ValueError(
                "Variáveis de ambiente de conexão não definidas corretamente.")

        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        engine = create_engine(connection_string)

        # Testa a conexão
        with engine.connect() as test_conn:
            test_conn.execute(text("SELECT 1"))

        return engine

    except Exception as e:
        logging.error(
            f"Erro ao criar engine para o banco {database}: {str(e)}")
        return None


@contextmanager
def get_engine(database: str):
    """Context manager para criação e descarte automático da engine.

    Args:
        database: Nome do banco de dados

    Yields:
        Engine SQLAlchemy

    Raises:
        RuntimeError: Se a conexão falhar
    """
    engine = criar_engine(database)
    if engine is None:
        raise RuntimeError(f"Falha ao conectar ao banco {database}")

    try:
        yield engine
    finally:
        engine.dispose()


def consultar_saldo_extra_diario_tipos(
    data_inicio: str = DEFAULT_DATA_INICIO,
    data_fim:    str = DEFAULT_DATA_FIM
) -> Optional[pd.DataFrame]:
    """Consulta saldo extra diário com cada plano como coluna (formato pivot).

    Gera planilha: Data | PINGOU | GRATUITO | SOMAPAY | TOTAL_DIA

    Args:
        data_inicio: Data inicial no formato 'YYYY-MM-DD'
        data_fim: Data final no formato 'YYYY-MM-DD'

    Returns:
        DataFrame pivotado com planos como colunas ou None em caso de falha
    """
    query_sql = f"""
        SELECT 
            dt_solicitacao::date as data,
            CASE 
                WHEN valor_solicitado <= 49.99 THEN 'PINGOU'
                WHEN valor_solicitado >= 50.00 AND valor_solicitado <= 100.00 THEN 'GRATUITO'
                WHEN valor_solicitado >= 100.01 THEN 'SOMAPAY'
                ELSE 'INDEFINIDO'
            END as plano_cliente,
            SUM(valor_solicitado) as valor_total
        
        FROM public.saldo_extra
        WHERE dt_solicitacao >= '{data_inicio}'::timestamp
            AND dt_solicitacao <= '{data_fim} 23:59:59'::timestamp
        
        GROUP BY dt_solicitacao::date, plano_cliente
        ORDER BY dt_solicitacao::date
    """

    try:
        with get_engine('bi') as engine, engine.connect() as conn:
            df = pd.read_sql(query_sql, conn)

            if df is not None and not df.empty:
                # Converte para datetime ANTES do pivot para ordenação correta
                df['data'] = pd.to_datetime(df['data'])

                # Cria tabela pivot: Data nas linhas, plano_cliente nas colunas
                df_pivot = df.pivot_table(
                    index='data',
                    columns='plano_cliente',
                    values='valor_total',
                    fill_value=0,
                    aggfunc='sum'
                ).round(2)

                # Reset index para ter Data como coluna
                df_pivot = df_pivot.reset_index()

                # AGORA formata a data para DD/MM/YYYY (após o pivot e ordenação)
                df_pivot['data'] = df_pivot['data'].dt.strftime('%d/%m/%Y')
                df_pivot = df_pivot.rename(columns={'data': 'Data'})

                # Adiciona coluna de total geral por linha
                colunas_planos = [
                    col for col in df_pivot.columns if col != 'Data']
                df_pivot['TOTAL_DIA'] = df_pivot[colunas_planos].sum(axis=1)

                # Reordena colunas: Data, planos ordenados, TOTAL_DIA por último
                planos_ordenados = [
                    'PINGOU', 'GRATUITO', 'SOMAPAY', 'INDEFINIDO']
                colunas_ordenadas = ['Data']

                # Adiciona planos na ordem desejada
                for plano in planos_ordenados:
                    if plano in df_pivot.columns:
                        colunas_ordenadas.append(plano)

                # Adiciona outros planos que possam existir
                for col in df_pivot.columns:
                    if col not in colunas_ordenadas and col != 'TOTAL_DIA':
                        colunas_ordenadas.append(col)

                # Adiciona total por último
                colunas_ordenadas.append('TOTAL_DIA')

                df_pivot = df_pivot[colunas_ordenadas]

                logging.info(
                    f"Consulta de saldo extra diário executada com sucesso. Dias: {len(df_pivot)}, Planos: {len(colunas_planos)}")

            return df_pivot

    except Exception as e:
        logging.error(
            f"Erro na consulta de saldo extra diário por planos: {str(e)}")
        return None


def consultar_saldo_extra_emprestado(
    data_inicio: str = DEFAULT_DATA_INICIO,
    data_fim:    str = DEFAULT_DATA_FIM
) -> Optional[pd.DataFrame]:
    """Consulta saldo extra emprestado por faixas de valor (Pingou, Gratuito, Somapay).

    Classificação:
    - Pingou: valores até R$ 49,99
    - Gratuito: valores de R$ 50,00 até R$ 100,00  
    - Somapay: valores acima de R$ 100,01

    Args:
        data_inicio: Data inicial no formato 'YYYY-MM-DD'
        data_fim: Data final no formato 'YYYY-MM-DD'

    Returns:
        DataFrame com análise de saldo extra por planos ou None em caso de falha
    """
    # Constroi a query com string formatting para evitar problemas de parâmetros
    query_sql = f"""
        WITH classificacao_planos AS (
            SELECT 
                *,
                CASE 
                    WHEN valor_solicitado <= 49.99 THEN 'Pingou'
                    WHEN valor_solicitado >= 50.00 AND valor_solicitado <= 100.00 THEN 'Gratuito'
                    WHEN valor_solicitado >= 100.01 THEN 'Somapay'
                    ELSE 'Indefinido'
                END as plano_cliente
                
            FROM public.saldo_extra
            WHERE dt_solicitacao >= '{data_inicio}'::timestamp
                AND dt_solicitacao <= '{data_fim} 23:59:59'::timestamp
        )
        
        SELECT 
            'RESUMO_GERAL' as categoria,
            'Total de Operações' as metrica,
            COUNT(*)::text as valor,
            'Número total de empréstimos no período' as descricao
        FROM classificacao_planos
        
        UNION ALL
        SELECT 'RESUMO_GERAL', 'Valor Total Solicitado', 'R$ ' || TO_CHAR(SUM(valor_solicitado), 'FM999,999,999,990.00'), 'Soma de todos os valores solicitados' FROM classificacao_planos
        UNION ALL
        SELECT 'RESUMO_GERAL', 'Valor Total Pago', 'R$ ' || TO_CHAR(SUM(valor_pago), 'FM999,999,999,990.00'), 'Soma de todos os valores pagos (principal + juros)' FROM classificacao_planos
        UNION ALL
        SELECT 'RESUMO_GERAL', 'Total de Juros Cobrados', 'R$ ' || TO_CHAR(SUM(valor_pago - valor_solicitado), 'FM999,999,999,990.00'), 'Diferença entre pago e solicitado' FROM classificacao_planos
        UNION ALL
        SELECT 'RESUMO_GERAL', 'Clientes Únicos', COUNT(DISTINCT cpf)::text, 'CPFs únicos que fizeram empréstimos' FROM classificacao_planos
        
        UNION ALL
        SELECT 'PINGOU', 'Operações', COUNT(*)::text, 'Empréstimos até R$ 49,99' FROM classificacao_planos WHERE plano_cliente = 'Pingou'
        UNION ALL
        SELECT 'PINGOU', 'Valor Solicitado', 'R$ ' || TO_CHAR(COALESCE(SUM(valor_solicitado), 0), 'FM999,999,999,990.00'), 'Total solicitado - Pingou' FROM classificacao_planos WHERE plano_cliente = 'Pingou'
        UNION ALL
        SELECT 'PINGOU', 'Valor Pago', 'R$ ' || TO_CHAR(COALESCE(SUM(valor_pago), 0), 'FM999,999,999,990.00'), 'Total pago - Pingou' FROM classificacao_planos WHERE plano_cliente = 'Pingou'
        UNION ALL
        SELECT 'PINGOU', 'Clientes Únicos', COUNT(DISTINCT cpf)::text, 'CPFs únicos - Pingou' FROM classificacao_planos WHERE plano_cliente = 'Pingou'
        UNION ALL
        SELECT 'PINGOU', 'Valor Médio', 'R$ ' || TO_CHAR(COALESCE(AVG(valor_solicitado), 0), 'FM999,999,999,990.00'), 'Valor médio por empréstimo - Pingou' FROM classificacao_planos WHERE plano_cliente = 'Pingou'
        
        UNION ALL
        SELECT 'GRATUITO', 'Operações', COUNT(*)::text, 'Empréstimos R$ 50,00 - R$ 100,00' FROM classificacao_planos WHERE plano_cliente = 'Gratuito'
        UNION ALL
        SELECT 'GRATUITO', 'Valor Solicitado', 'R$ ' || TO_CHAR(COALESCE(SUM(valor_solicitado), 0), 'FM999,999,999,990.00'), 'Total solicitado - Gratuito' FROM classificacao_planos WHERE plano_cliente = 'Gratuito'
        UNION ALL
        SELECT 'GRATUITO', 'Valor Pago', 'R$ ' || TO_CHAR(COALESCE(SUM(valor_pago), 0), 'FM999,999,999,990.00'), 'Total pago - Gratuito' FROM classificacao_planos WHERE plano_cliente = 'Gratuito'
        UNION ALL
        SELECT 'GRATUITO', 'Clientes Únicos', COUNT(DISTINCT cpf)::text, 'CPFs únicos - Gratuito' FROM classificacao_planos WHERE plano_cliente = 'Gratuito'
        UNION ALL
        SELECT 'GRATUITO', 'Valor Médio', 'R$ ' || TO_CHAR(COALESCE(AVG(valor_solicitado), 0), 'FM999,999,999,990.00'), 'Valor médio por empréstimo - Gratuito' FROM classificacao_planos WHERE plano_cliente = 'Gratuito'
        
        UNION ALL
        SELECT 'SOMAPAY', 'Operações', COUNT(*)::text, 'Empréstimos acima de R$ 100,01' FROM classificacao_planos WHERE plano_cliente = 'Somapay'
        UNION ALL
        SELECT 'SOMAPAY', 'Valor Solicitado', 'R$ ' || TO_CHAR(COALESCE(SUM(valor_solicitado), 0), 'FM999,999,999,990.00'), 'Total solicitado - Somapay' FROM classificacao_planos WHERE plano_cliente = 'Somapay'
        UNION ALL
        SELECT 'SOMAPAY', 'Valor Pago', 'R$ ' || TO_CHAR(COALESCE(SUM(valor_pago), 0), 'FM999,999,999,990.00'), 'Total pago - Somapay' FROM classificacao_planos WHERE plano_cliente = 'Somapay'
        UNION ALL
        SELECT 'SOMAPAY', 'Clientes Únicos', COUNT(DISTINCT cpf)::text, 'CPFs únicos - Somapay' FROM classificacao_planos WHERE plano_cliente = 'Somapay'
        UNION ALL
        SELECT 'SOMAPAY', 'Valor Médio', 'R$ ' || TO_CHAR(COALESCE(AVG(valor_solicitado), 0), 'FM999,999,999,990.00'), 'Valor médio por empréstimo - Somapay' FROM classificacao_planos WHERE plano_cliente = 'Somapay'
        
        UNION ALL
        SELECT 'SITUACAO_FUNCIONARIO', 'Funcionários Ativos', COUNT(*)::text, 'Empréstimos de funcionários ativos' FROM classificacao_planos WHERE UPPER(situacao_fnc) = 'ATIVO'
        UNION ALL
        SELECT 'SITUACAO_FUNCIONARIO', 'Funcionários Demitidos', COUNT(*)::text, 'Empréstimos de funcionários demitidos' FROM classificacao_planos WHERE UPPER(situacao_fnc) = 'DEMITIDO'
        
        UNION ALL
        SELECT 'CANAIS', 'ATM Próprio', COUNT(*)::text, 'Operações via ATM Próprio' FROM classificacao_planos WHERE UPPER(canal) = 'ATM PROPRIO'
        UNION ALL
        SELECT 'CANAIS', 'Fácil', COUNT(*)::text, 'Operações via canal Fácil' FROM classificacao_planos WHERE UPPER(canal) = 'FACIL'
        
        ORDER BY categoria, metrica
    """

    try:
        with get_engine('bi') as engine, engine.connect() as conn:
            df = pd.read_sql(query_sql, conn)
            logging.info(
                f"Consulta de saldo extra executada com sucesso para período {data_inicio} a {data_fim}. Métricas encontradas: {len(df) if df is not None else 0}")
            return df

    except Exception as e:
        logging.error(f"Erro na consulta de saldo extra: {str(e)}")
        return None


def consultar_distribuicao_por_tipo_e_data(
    data_inicio: str = DEFAULT_DATA_INICIO,
    data_fim:    str = DEFAULT_DATA_FIM
) -> Optional[pd.DataFrame]:
    """Consulta distribuição de valores pagos por tipo e data.

    Gera planilha detalhada: Data | Type_Payment | Valor (R$) | Quantidade

    Args:
        data_inicio: Data inicial no formato 'YYYY-MM-DD'
        data_fim: Data final no formato 'YYYY-MM-DD'

    Returns:
        DataFrame com distribuição por tipo e data ou None em caso de falha
    """
    query = text("""
        SELECT 
            pf.import_date::date as data,
            pf.type_payment,
            SUM(pfl.value_payment) as valor_total,
            COUNT(pfl.id) as quantidade_pagamentos,
            COUNT(DISTINCT pfl.cpf) as funcionarios_distintos,
            AVG(pfl.value_payment) as valor_medio,
            MIN(pfl.value_payment) as valor_minimo,
            MAX(pfl.value_payment) as valor_maximo
        
        FROM public.payment_file_lines pfl
        INNER JOIN public.payments_files pf ON pfl.file_id = pf.id
        
        WHERE pf.import_date >= :data_inicio
            AND pf.import_date <= :data_fim
            AND pfl.processing_status = 'PAID'
        
        GROUP BY pf.import_date::date, pf.type_payment
        ORDER BY pf.import_date::date, valor_total DESC
    """).bindparams(data_inicio=data_inicio, data_fim=data_fim)

    try:
        with get_engine('companies') as engine, engine.connect() as conn:
            df = pd.read_sql(query, conn)

            if df is not None and not df.empty:
                # Renomeia colunas para melhor legibilidade
                df = df.rename(columns={
                    'data': 'Data',
                    'type_payment': 'Tipo_Pagamento',
                    'valor_total': 'Valor (R$)',
                    'quantidade_pagamentos': 'Quantidade',
                    'funcionarios_distintos': 'Funcionários',
                    'valor_medio': 'Valor_Médio (R$)',
                    'valor_minimo': 'Valor_Mín (R$)',
                    'valor_maximo': 'Valor_Máx (R$)'
                })

                # Formata a data para DD/MM/YYYY
                df['Data'] = pd.to_datetime(df['Data']).dt.strftime('%d/%m/%Y')

                # Formata valores monetários
                df['Valor (R$)'] = df['Valor (R$)'].round(2)
                df['Valor_Médio (R$)'] = df['Valor_Médio (R$)'].round(2)
                df['Valor_Mín (R$)'] = df['Valor_Mín (R$)'].round(2)
                df['Valor_Máx (R$)'] = df['Valor_Máx (R$)'].round(2)

                logging.info(
                    f"Consulta de distribuição por tipo e data executada com sucesso para período {data_inicio} a {data_fim}. Registros encontrados: {len(df)}")

            return df

    except Exception as e:
        logging.error(
            f"Erro na consulta de distribuição por tipo e data: {str(e)}")
        return None


def consultar_resumo_diario_tipos(
    data_inicio: str = DEFAULT_DATA_INICIO,
    data_fim:    str = DEFAULT_DATA_FIM
) -> Optional[pd.DataFrame]:
    """Consulta resumo diário com cada tipo como coluna (formato pivot).

    Gera planilha: Data | SALARY | VACATION | RESCISSION | ... (cada tipo como coluna)

    Args:
        data_inicio: Data inicial no formato 'YYYY-MM-DD'
        data_fim: Data final no formato 'YYYY-MM-DD'

    Returns:
        DataFrame pivotado com tipos como colunas ou None em caso de falha
    """
    query = text("""
        SELECT 
            pf.import_date::date as data,
            pf.type_payment,
            SUM(pfl.value_payment) as valor_total
        
        FROM public.payment_file_lines pfl
        INNER JOIN public.payments_files pf ON pfl.file_id = pf.id
        
        WHERE pf.import_date >= :data_inicio
            AND pf.import_date <= :data_fim
            AND pfl.processing_status = 'PAID'
        
        GROUP BY pf.import_date::date, pf.type_payment
        ORDER BY pf.import_date::date
    """).bindparams(data_inicio=data_inicio, data_fim=data_fim)

    try:
        with get_engine('companies') as engine, engine.connect() as conn:
            df = pd.read_sql(query, conn)

            if df is not None and not df.empty:
                # Converte para datetime ANTES do pivot para ordenação correta
                df['data'] = pd.to_datetime(df['data'])

                # Cria tabela pivot: Data nas linhas, type_payment nas colunas
                df_pivot = df.pivot_table(
                    index='data',
                    columns='type_payment',
                    values='valor_total',
                    fill_value=0,
                    aggfunc='sum'
                ).round(2)

                # Reset index para ter Data como coluna
                df_pivot = df_pivot.reset_index()

                # AGORA formata a data para DD/MM/YYYY (após o pivot e ordenação)
                df_pivot['data'] = df_pivot['data'].dt.strftime('%d/%m/%Y')
                df_pivot = df_pivot.rename(columns={'data': 'Data'})

                # Adiciona coluna de total geral por linha
                colunas_valores = [
                    col for col in df_pivot.columns if col != 'Data']
                df_pivot['TOTAL_DIA'] = df_pivot[colunas_valores].sum(axis=1)

                # Reordena colunas: Data, tipos principais primeiro, depois outros, TOTAL_DIA por último
                tipos_principais = ['SALARY', 'VACATION',
                                    'ADVANCE_SALARY', 'RESCISSION', 'DEPOSIT']
                colunas_ordenadas = ['Data']

                # Adiciona tipos principais primeiro
                for tipo in tipos_principais:
                    if tipo in df_pivot.columns:
                        colunas_ordenadas.append(tipo)

                # Adiciona outros tipos
                for col in df_pivot.columns:
                    if col not in colunas_ordenadas and col != 'TOTAL_DIA':
                        colunas_ordenadas.append(col)

                # Adiciona total por último
                colunas_ordenadas.append('TOTAL_DIA')

                df_pivot = df_pivot[colunas_ordenadas]

                logging.info(
                    f"Consulta de resumo diário tipos executada com sucesso. Dias: {len(df_pivot)}, Tipos: {len(colunas_valores)}")

            return df_pivot

    except Exception as e:
        logging.error(f"Erro na consulta de resumo diário por tipos: {str(e)}")
        return None


def _criar_diretorio_se_nexiste(caminho: str) -> None:
    """Cria diretórios necessários para o caminho especificado."""
    diretorio = os.path.dirname(caminho)
    if diretorio and not os.path.exists(diretorio):
        os.makedirs(diretorio, exist_ok=True)
        logging.info(f"Diretório criado: {diretorio}")


def salvar_em_excel(df: pd.DataFrame, caminho_arquivo: str, nome_aba: str = "base_historica") -> bool:
    """Salva o DataFrame em um arquivo Excel.

    Args:
        df: DataFrame a ser salvo
        caminho_arquivo: Caminho completo do arquivo

    Returns:
        True se salvou com sucesso, False caso contrário
    """
    try:
        _criar_diretorio_se_nexiste(caminho_arquivo)
        df.to_excel(caminho_arquivo, index=False, sheet_name=nome_aba)
        logging.info(f"Excel salvo em: {caminho_arquivo}")
        return True

    except PermissionError as e:
        logging.error(f"Permissão negada para salvar Excel: {str(e)}")
    except Exception as e:
        logging.error(f"Erro ao salvar Excel: {str(e)}")

    return False


def salvar_em_csv(df: pd.DataFrame, caminho_arquivo: str) -> bool:
    """Salva o DataFrame em um arquivo CSV.

    Args:
        df: DataFrame a ser salvo
        caminho_arquivo: Caminho completo do arquivo

    Returns:
        True se salvou com sucesso, False caso contrário
    """
    try:
        _criar_diretorio_se_nexiste(caminho_arquivo)
        df.to_csv(caminho_arquivo, index=False)
        logging.info(f"CSV salvo em: {caminho_arquivo}")
        return True

    except PermissionError as e:
        logging.error(f"Permissão negada para salvar CSV: {str(e)}")
    except Exception as e:
        logging.error(f"Erro ao salvar CSV: {str(e)}")

    return False


def gerar_nomes_arquivos() -> Tuple[str, str, str, str, str, str, str, str]:
    """Gera nomes de arquivos com timestamp para Excel e CSV."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')

    # Garante que o diretório base existe
    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    return (
        os.path.join(RESULTADOS_DIR,
                     f"distribuicao_tipos_detalhada_{timestamp}.xlsx"),
        os.path.join(RESULTADOS_DIR,
                     f"distribuicao_tipos_detalhada_{timestamp}.csv"),
        os.path.join(RESULTADOS_DIR, f"base_historica.xlsx"),
        os.path.join(RESULTADOS_DIR, f"base_historica.csv"),
        os.path.join(RESULTADOS_DIR,
                     f"saldo_extra_emprestado_{timestamp}.xlsx"),
        os.path.join(RESULTADOS_DIR,
                     f"saldo_extra_emprestado_{timestamp}.csv"),
        os.path.join(RESULTADOS_DIR,
                     f"saldo_extra_diario_planos_{timestamp}.xlsx"),
        os.path.join(RESULTADOS_DIR,
                     f"saldo_extra_diario_planos_{timestamp}.csv")
    )


def inserir_emprestimo_no_resumo(df_resumo_diario: pd.DataFrame, df_saldo_extra_diario: pd.DataFrame) -> pd.DataFrame:
    """
    Insere a coluna 'TOTAL_DIA' do saldo extra diário no resumo diário, 
    entre as colunas 'Data' e 'SALARY', renomeando para 'EMPRESTIMO'.
    """
    # Garante que as datas estão no mesmo formato
    df_resumo = df_resumo_diario.copy()
    df_saldo = df_saldo_extra_diario[['Data', 'TOTAL_DIA']].copy()
    df_saldo = df_saldo.rename(columns={'TOTAL_DIA': 'EMPRESTIMO'})
    # Faz o merge pela coluna Data
    df_merged = pd.merge(df_resumo, df_saldo, on='Data', how='left')
    # Move a coluna EMPRESTIMO para logo após Data
    cols = df_merged.columns.tolist()
    if 'EMPRESTIMO' in cols and 'Data' in cols:
        cols.insert(cols.index('Data') + 1, cols.pop(cols.index('EMPRESTIMO')))
    # Opcional: garantir que SALARY existe antes de ordenar
    df_merged = df_merged[cols]
    return df_merged


def main():
    """Fluxo principal de execução."""
    logging.info(
        "Iniciando análise de distribuição de pagamentos por tipo e data + saldo extra emprestado.")

    # Primeira consulta: Distribuição detalhada por tipo e data
    df_distribuicao_detalhada = consultar_distribuicao_por_tipo_e_data()
    if df_distribuicao_detalhada is None or df_distribuicao_detalhada.empty:
        logging.error(
            "Consulta de distribuição detalhada não retornou dados válidos.")
        return

    # Segunda consulta: Resumo diário com tipos como colunas
    df_resumo_diario = consultar_resumo_diario_tipos()
    if df_resumo_diario is None or df_resumo_diario.empty:
        logging.error("Consulta de resumo diário não retornou dados válidos.")
        return

    # Terceira consulta: Saldo extra emprestado
    df_saldo_extra = consultar_saldo_extra_emprestado()
    if df_saldo_extra is None or df_saldo_extra.empty:
        logging.error("Consulta de saldo extra não retornou dados válidos.")
        return

    # Quarta consulta: Saldo extra diário por planos
    df_saldo_extra_diario = consultar_saldo_extra_diario_tipos()
    if df_saldo_extra_diario is None or df_saldo_extra_diario.empty:
        logging.error(
            "Consulta de saldo extra diário não retornou dados válidos.")
        return

    logging.info(f"Dados recuperados com sucesso.")

    # Adicione esta linha para criar o DataFrame com a coluna EMPRESTIMO
    df_resumo_diario_com_emprestimo = inserir_emprestimo_no_resumo(
        df_resumo_diario, df_saldo_extra_diario)

    # Exibe análise de saldo extra
    print(f"\n=== ANÁLISE SALDO EXTRA EMPRESTADO ===")
    if len(df_saldo_extra) > 0:
        for _, row in df_saldo_extra.iterrows():
            print(f"[{row['categoria']}] {row['metrica']}: {row['valor']}")

    # Exibe amostra do saldo extra diário
    print(f"\n=== SALDO EXTRA DIÁRIO POR PLANOS (AMOSTRA) ===")
    if len(df_saldo_extra_diario) > 0:
        print(f"Dias com empréstimos: {len(df_saldo_extra_diario)}")
        # -2 por Data e TOTAL_DIA
        print(f"Planos encontrados: {len(df_saldo_extra_diario.columns) - 2}")
        print(f"Primeiros 10 dias:")

        # Mostra todas as colunas de planos + total
        colunas_exibir = df_saldo_extra_diario.columns.tolist()
        print(df_saldo_extra_diario[colunas_exibir].head(
            10).to_string(index=False))

        # Estatísticas dos planos
        colunas_planos = [col for col in df_saldo_extra_diario.columns if col not in [
            'Data', 'TOTAL_DIA']]
        if colunas_planos:
            volumes_por_plano = df_saldo_extra_diario[colunas_planos].sum(
            ).sort_values(ascending=False)
            print(f"\n💰 Volume Total por Plano:")
            for plano, valor in volumes_por_plano.items():
                if valor > 0:
                    print(f"   {plano}: R$ {valor:,.2f}")

            # Dia com maior volume
            if 'TOTAL_DIA' in df_saldo_extra_diario.columns:
                maior_dia_idx = df_saldo_extra_diario['TOTAL_DIA'].idxmax()
                maior_dia_data = df_saldo_extra_diario.loc[maior_dia_idx, 'Data']
                maior_dia_valor = df_saldo_extra_diario.loc[maior_dia_idx, 'TOTAL_DIA']
                print(
                    f"\n📈 Maior dia de empréstimos: {maior_dia_data} (R$ {maior_dia_valor:,.2f})")

    # Exibe amostra da distribuição detalhada
    print(f"\n=== DISTRIBUIÇÃO POR TIPO E DATA (AMOSTRA) ===")
    if len(df_distribuicao_detalhada) > 0:
        print(
            f"Total de registros tipo/data: {len(df_distribuicao_detalhada)}")
        print(f"Primeiros 15 registros:")
        # Mostra apenas colunas principais para não sobrecarregar
        colunas_principais = ['Data', 'Tipo_Pagamento',
                              'Valor (R$)', 'Quantidade', 'Funcionários']
        print(df_distribuicao_detalhada[colunas_principais].head(
            15).to_string(index=False))

        if len(df_distribuicao_detalhada) > 15:
            print(
                f"\n... e mais {len(df_distribuicao_detalhada) - 15} registros")

        # Estatísticas gerais
        total_valor = df_distribuicao_detalhada['Valor (R$)'].sum()
        tipos_unicos = df_distribuicao_detalhada['Tipo_Pagamento'].nunique()
        datas_unicas = df_distribuicao_detalhada['Data'].nunique()

        print(f"\n📊 Estatísticas Gerais:")
        print(f"   Total processado: R$ {total_valor:,.2f}")
        print(f"   Tipos de pagamento: {tipos_unicos}")
        print(f"   Dias com movimentação: {datas_unicas}")

    # Exibe amostra do resumo diário
    print(f"\n=== RESUMO DIÁRIO POR TIPOS (AMOSTRA) ===")
    if len(df_resumo_diario) > 0:
        print(f"Dias com movimentação: {len(df_resumo_diario)}")
        # -2 por Data e TOTAL_DIA
        print(
            f"Tipos de pagamento encontrados: {len(df_resumo_diario.columns) - 2}")
        print(f"Primeiros 10 dias:")

        # Mostra apenas primeiras colunas para não sobrecarregar a exibição
        # Data + 5 primeiros tipos
        colunas_exibir = df_resumo_diario.columns[:6].tolist()
        if 'TOTAL_DIA' in df_resumo_diario.columns:
            colunas_exibir.append('TOTAL_DIA')

        print(df_resumo_diario[colunas_exibir].head(10).to_string(index=False))

        if len(df_resumo_diario.columns) > 7:  # Se tem mais colunas além das 6 mostradas
            print(
                f"\n   (Planilha completa contém todas as {len(df_resumo_diario.columns)-1} colunas de tipos)")

        # Top 3 tipos por volume
        colunas_tipos = [
            col for col in df_resumo_diario.columns if col not in ['Data', 'TOTAL_DIA']]
        if colunas_tipos:
            volumes_por_tipo = df_resumo_diario[colunas_tipos].sum(
            ).sort_values(ascending=False)
            print(f"\n🏆 Top 3 Tipos por Volume Total:")
            for i, (tipo, valor) in enumerate(volumes_por_tipo.head(3).items(), 1):
                print(f"   {i}. {tipo}: R$ {valor:,.2f}")

    # Salva os resultados
    excel_dist, csv_dist, excel_res, csv_res, excel_saldo, csv_saldo, excel_saldo_diario, csv_saldo_diario = gerar_nomes_arquivos()
    # Salve com aba base_historica
    sucesso_dist = salvar_em_excel(df_distribuicao_detalhada, excel_dist, "base_historica") and salvar_em_csv(
        df_distribuicao_detalhada, csv_dist)
    # ALTERE AQUI: use o DataFrame com EMPRESTIMO
    sucesso_res = salvar_em_excel(
        df_resumo_diario_com_emprestimo, excel_res, "base_historica") and salvar_em_csv(df_resumo_diario_com_emprestimo, csv_res)
    sucesso_saldo = salvar_em_excel(
        df_saldo_extra, excel_saldo) and salvar_em_csv(df_saldo_extra, csv_saldo)
    sucesso_saldo_diario = salvar_em_excel(df_saldo_extra_diario, excel_saldo_diario) and salvar_em_csv(
        df_saldo_extra_diario, csv_saldo_diario)

    if sucesso_dist and sucesso_res and sucesso_saldo and sucesso_saldo_diario:
        logging.info(
            "Exportação de todos os relatórios concluída com sucesso!")
        print(f"\n🎯 RESUMO EXECUTIVO:")
        print(
            f"📊 Distribuição Detalhada: {len(df_distribuicao_detalhada)} registros tipo/data")
        print(
            f"📋 Resumo Diário por Tipos: {len(df_resumo_diario)} dias × {len(df_resumo_diario.columns)-1} tipos")
        print(f"💰 Saldo Extra: {len(df_saldo_extra)} métricas de empréstimos")
        print(
            f"📅 Saldo Extra Diário: {len(df_saldo_extra_diario)} dias × {len(df_saldo_extra_diario.columns)-1} planos")
        print(f"📁 Arquivos salvos em: {RESULTADOS_DIR}")

        print(f"\n🗂️ ARQUIVOS GERADOS:")
        print(f"   1. Distribuição detalhada (cada tipo/data com estatísticas completas)")
        print(f"   2. Resumo diário pivot (cada tipo como coluna para análise rápida)")
        print(
            f"   3. Saldo extra emprestado (análise por planos: Pingou, Gratuito, Somapay)")
        print(f"   4. 🆕 Saldo extra diário pivot (cada plano como coluna por data)")
    else:
        logging.error("Ocorreram erros durante a exportação.")


if __name__ == '__main__':
    main()
