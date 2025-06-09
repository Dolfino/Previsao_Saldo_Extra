#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validador Simples do Processo SARIMAX
====================================

Validador simplificado sem erros de sintaxe para verificar o processo.

Uso: python validador_simples.py
"""

import json
import os
from datetime import datetime


def verificar_arquivos():
    """Verifica se os arquivos necessários estão presentes"""
    print("🔍 VERIFICANDO ARQUIVOS NECESSÁRIOS")
    print("=" * 50)

    arquivos = {
        'Analise_Sarimax_normal_v01_02_OK.py': 'Script principal do modelo',
        'dashboard_integration.py': 'Script de integração',
        'base_historica.csv': 'Dados históricos',
        'dashboard.html': 'Interface do dashboard'
    }

    encontrados = 0
    total = len(arquivos)

    for arquivo, descricao in arquivos.items():
        existe = os.path.exists(arquivo)
        status = "✅" if existe else "❌"
        print(f"   {status} {arquivo} - {descricao}")
        if existe:
            encontrados += 1

    print(f"\n📊 Resumo: {encontrados}/{total} arquivos encontrados")
    return encontrados == total


def verificar_csv():
    """Verifica o formato do CSV"""
    print("\n📊 VERIFICANDO CSV BASE")
    print("=" * 50)

    if not os.path.exists('base_historica.csv'):
        print("   ❌ base_historica.csv não encontrado")
        return False

    try:
        with open('base_historica.csv', 'r', encoding='utf-8') as f:
            primeiras_linhas = [f.readline().strip() for _ in range(3)]

        print("   📋 Primeiras 3 linhas:")
        for i, linha in enumerate(primeiras_linhas, 1):
            print(f"      {i}. {linha[:80]}{'...' if len(linha) > 80 else ''}")

        # Verificar separador
        header = primeiras_linhas[0] if primeiras_linhas else ""
        tem_ponto_virgula = ';' in header

        print(
            f"\n   🔍 Separador ';': {'✅ Encontrado' if tem_ponto_virgula else '❌ Não encontrado'}")

        # Verificar colunas esperadas
        colunas_esperadas = ['Data', 'EMPRESTIMO', 'SALARY', 'RESCISSION']
        for coluna in colunas_esperadas:
            tem_coluna = coluna.upper() in header.upper()
            status = "✅" if tem_coluna else "❌"
            print(f"   {status} Coluna {coluna}")

        return tem_ponto_virgula

    except Exception as e:
        print(f"   ❌ Erro ao ler CSV: {e}")
        return False


def verificar_outputs_modelo():
    """Verifica se o modelo gerou os arquivos esperados"""
    print("\n🚀 VERIFICANDO OUTPUTS DO MODELO")
    print("=" * 50)

    outputs_esperados = {
        'sarimax_analysis.log': 'Log de execução',
        'modelo_sarimax_melhorado.joblib': 'Modelo treinado',
        'dashboard_data.json': 'Dados para dashboard'
    }

    encontrados = 0
    total = len(outputs_esperados)

    for arquivo, descricao in outputs_esperados.items():
        existe = os.path.exists(arquivo)
        status = "✅" if existe else "❌"
        print(f"   {status} {arquivo} - {descricao}")

        if existe:
            encontrados += 1
            # Mostrar tamanho do arquivo
            try:
                tamanho = os.path.getsize(arquivo) / 1024  # KB
                print(f"      💾 Tamanho: {tamanho:.1f} KB")
            except:
                pass

    print(f"\n📊 Outputs gerados: {encontrados}/{total}")
    return encontrados == total


def verificar_json_dashboard():
    """Verifica se o JSON do dashboard está válido"""
    print("\n📊 VERIFICANDO dashboard_data.json")
    print("=" * 50)

    if not os.path.exists('dashboard_data.json'):
        print("   ❌ dashboard_data.json não encontrado")
        return False

    try:
        with open('dashboard_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        print("   ✅ JSON válido")

        # Verificar seções principais
        secoes = ['timestamp', 'metricas',
                  'features', 'modelo', 'configuracao']

        print("\n   🔍 Seções encontradas:")
        for secao in secoes:
            existe = secao in data
            status = "✅" if existe else "❌"
            print(f"      {status} {secao}")

        # Mostrar timestamp se disponível
        if 'timestamp' in data:
            try:
                timestamp = data['timestamp']
                print(f"\n   ⏰ Gerado em: {timestamp}")
            except:
                pass

        # Verificar métricas se disponíveis
        if 'metricas' in data and isinstance(data['metricas'], dict):
            print(f"\n   📈 Métricas encontradas: {len(data['metricas'])}")
            for metrica, valores in data['metricas'].items():
                if isinstance(valores, dict) and 'atual' in valores:
                    atual = valores['atual']
                    print(f"      • {metrica.upper()}: {atual}")

        # Verificar features se disponíveis
        if 'features' in data and isinstance(data['features'], list):
            num_features = len(data['features'])
            print(f"\n   🎯 Features selecionadas: {num_features}")

            # Mostrar top 3
            for i, feature in enumerate(data['features'][:3], 1):
                if isinstance(feature, dict) and 'name' in feature:
                    nome = feature['name']
                    importancia = feature.get('importance', 'N/A')
                    print(f"      {i}. {nome} (imp: {importancia})")

        return True

    except json.JSONDecodeError as e:
        print(f"   ❌ JSON inválido: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Erro ao verificar JSON: {e}")
        return False


def verificar_logs():
    """Verifica informações nos logs"""
    print("\n📋 VERIFICANDO LOGS DE EXECUÇÃO")
    print("=" * 50)

    if not os.path.exists('sarimax_analysis.log'):
        print("   ❌ Log não encontrado")
        return False

    try:
        with open('sarimax_analysis.log', 'r', encoding='utf-8') as f:
            conteudo = f.read()

        # Procurar indicadores chave
        indicadores = {
            'ANÁLISE COMPLETA': 'Execução finalizada',
            'RMSE': 'Métrica RMSE calculada',
            'MAE': 'Métrica MAE calculada',
            'R²': 'Métrica R² calculada',
            'features selecionadas': 'Features foram selecionadas',
            'dashboard_data.json': 'JSON foi exportado'
        }

        print("   🔍 Indicadores no log:")
        for indicador, descricao in indicadores.items():
            encontrado = indicador.lower() in conteudo.lower()
            status = "✅" if encontrado else "❌"
            print(f"      {status} {descricao}")

        # Tentar extrair valores de métricas
        print("\n   📊 Tentando extrair métricas dos logs:")
        linhas = conteudo.split('\n')

        for linha in linhas:
            if 'RMSE:' in linha and '±' in linha:
                print(f"      📈 {linha.strip()}")
            elif 'MAE:' in linha and '±' in linha:
                print(f"      📈 {linha.strip()}")
            elif 'R²:' in linha and '±' in linha:
                print(f"      📈 {linha.strip()}")

        return True

    except Exception as e:
        print(f"   ❌ Erro ao ler log: {e}")
        return False


def gerar_relatorio_simples():
    """Gera relatório final simples"""
    print("\n" + "=" * 60)
    print("📋 RELATÓRIO FINAL DE VALIDAÇÃO")
    print("=" * 60)

    # Executar todas as verificações
    resultados = {
        'arquivos_presentes': verificar_arquivos(),
        'csv_valido': verificar_csv(),
        'outputs_gerados': verificar_outputs_modelo(),
        'json_valido': verificar_json_dashboard(),
        'logs_ok': verificar_logs()
    }

    # Calcular score geral
    total_checks = len(resultados)
    checks_ok = sum(resultados.values())
    score = (checks_ok / total_checks) * 100

    print(f"\n🎯 SCORE GERAL: {score:.1f}%")
    print(f"✅ Verificações OK: {checks_ok}/{total_checks}")

    # Status final
    if score == 100:
        print("\n🎉 PROCESSO VALIDADO COM SUCESSO!")
        print("✅ Todos os checks passaram")
        print("🚀 Dashboard pode ser usado com confiança")

        print("\n🔄 PRÓXIMOS PASSOS:")
        print("   1. Abra dashboard.html no navegador")
        print("   2. Clique em '🔄 Carregar Dados do JSON'")
        print("   3. Explore as métricas e visualizações")

    elif score >= 80:
        print("\n⚠️ PROCESSO PARCIALMENTE VALIDADO")
        print("✅ Maioria dos checks passou")
        print("🔧 Alguns ajustes podem ser necessários")

    else:
        print("\n❌ PROBLEMAS DETECTADOS")
        print("🔧 Processo precisa ser executado/corrigido")

        print("\n🔄 AÇÕES RECOMENDADAS:")
        if not resultados['arquivos_presentes']:
            print("   📁 Verificar se todos os arquivos estão na pasta")
        if not resultados['csv_valido']:
            print("   📊 Verificar formato do CSV (separador ';', decimal ',')")
        if not resultados['outputs_gerados']:
            print("   🚀 Executar: python Analise_Sarimax_normal_v01_02_OK.py")
        if not resultados['json_valido']:
            print("   📊 Verificar se modelo executou corretamente")

    # Salvar relatório
    relatorio = {
        'timestamp': datetime.now().isoformat(),
        'score': score,
        'resultados': resultados
    }

    try:
        with open('relatorio_validacao_simples.json', 'w', encoding='utf-8') as f:
            json.dump(relatorio, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Relatório salvo em: relatorio_validacao_simples.json")
    except Exception as e:
        print(f"\n⚠️ Erro ao salvar relatório: {e}")

    return score >= 80


def main():
    """Função principal"""
    print("🎯 VALIDADOR SIMPLES - PROCESSO SARIMAX")
    print("=" * 50)
    print("Verificação rápida do processo de execução\n")

    sucesso = gerar_relatorio_simples()
    return sucesso


if __name__ == "__main__":
    resultado = main()
    exit(0 if resultado else 1)
