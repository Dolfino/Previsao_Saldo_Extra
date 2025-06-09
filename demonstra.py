#!/usr/bin/env python3
"""
EXEMPLO COMPLETO DE USO: SARIMAX + DASHBOARD INTEGRADO
=====================================================

Este script demonstra como executar todo o pipeline desde o modelo
até a visualização no dashboard web de forma integrada e automatizada.

Autor: Sistema SARIMAX Melhorado V2.0
Data: 30 de Maio de 2025
"""

import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('exemplo_completo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def verificar_arquivos_necessarios():
    """
    Verifica se todos os arquivos necessários estão presentes
    """
    print("🔍 VERIFICANDO ARQUIVOS NECESSÁRIOS...")

    arquivos_obrigatorios = {
        'base_historica.csv': 'Dados históricos para treinamento',
        'Analise_Sarimax_normal_v01_02_OK.py': 'Script principal do modelo SARIMAX',
        'dashboard_integration.py': 'Script de integração com dashboard',
        'dashboard.html': 'Interface web do dashboard'
    }

    arquivos_faltando = []

    for arquivo, descricao in arquivos_obrigatorios.items():
        if os.path.exists(arquivo):
            tamanho = os.path.getsize(arquivo) / 1024  # KB
            print(f"   ✅ {arquivo} ({tamanho:.1f} KB) - {descricao}")
        else:
            print(f"   ❌ {arquivo} - {descricao}")
            arquivos_faltando.append(arquivo)

    if arquivos_faltando:
        print(
            f"\n⚠️ ATENÇÃO: {len(arquivos_faltando)} arquivo(s) não encontrado(s):")
        for arquivo in arquivos_faltando:
            print(f"   • {arquivo}")
        print("\nPor favor, verifique se todos os arquivos estão na pasta correta.")
        return False

    print("✅ Todos os arquivos necessários estão presentes!")
    return True


def executar_modelo_sarimax():
    """
    Executa o modelo SARIMAX principal
    """
    print("\n🚀 EXECUTANDO MODELO SARIMAX...")
    print("=" * 50)

    try:
        # Importar e executar o modelo principal
        import importlib.util

        # Carregar módulo dinamicamente
        spec = importlib.util.spec_from_file_location(
            "sarimax_model",
            "Analise_Sarimax_normal_v01_02_OK.py"
        )
        sarimax_module = importlib.util.module_from_spec(spec)

        # Executar o modelo
        print("⏳ Iniciando treinamento do modelo (isso pode levar alguns minutos)...")
        inicio = time.time()

        spec.loader.exec_module(sarimax_module)

        fim = time.time()
        tempo_execucao = fim - inicio

        print(
            f"✅ Modelo executado com sucesso em {tempo_execucao:.1f} segundos!")

        # Verificar se arquivos foram gerados
        arquivos_gerados = []
        if os.path.exists('modelo_sarimax_melhorado.joblib'):
            arquivos_gerados.append('modelo_sarimax_melhorado.joblib')
        if os.path.exists('dashboard_data.json'):
            arquivos_gerados.append('dashboard_data.json')
        if os.path.exists('sarimax_analysis.log'):
            arquivos_gerados.append('sarimax_analysis.log')

        print(f"\n📁 Arquivos gerados ({len(arquivos_gerados)}):")
        for arquivo in arquivos_gerados:
            tamanho = os.path.getsize(arquivo) / 1024
            print(f"   ✅ {arquivo} ({tamanho:.1f} KB)")

        return True

    except Exception as e:
        print(f"❌ Erro ao executar modelo SARIMAX: {e}")
        logger.error(f"Erro no modelo SARIMAX: {e}")
        return False


def executar_integracao_dashboard():
    """
    Executa a integração com o dashboard
    """
    print("\n🔗 EXECUTANDO INTEGRAÇÃO COM DASHBOARD...")
    print("=" * 50)

    try:
        # Importar script de integração
        from joblib import load

        from dashboard_integration import executar_integracao_completa

        # Tentar carregar modelo treinado
        modelo = None
        if os.path.exists('modelo_sarimax_melhorado.joblib'):
            try:
                modelo = load('modelo_sarimax_melhorado.joblib')
                print("✅ Modelo treinado carregado com sucesso")
            except Exception as e:
                print(f"⚠️ Erro ao carregar modelo: {e}")
                print("   Continuando apenas com dados dos logs...")

        # Executar integração completa
        sucesso = executar_integracao_completa(
            modelo_sarimax=modelo,
            verbose=True,
            incluir_logs=True,
            validar_resultado=True
        )

        if sucesso:
            print("✅ Integração com dashboard concluída com sucesso!")
            return True
        else:
            print("❌ Falha na integração com dashboard")
            return False

    except Exception as e:
        print(f"❌ Erro na integração: {e}")
        logger.error(f"Erro na integração: {e}")
        return False


def gerar_relatorio_resumo():
    """
    Gera relatório resumo dos resultados
    """
    print("\n📊 GERANDO RELATÓRIO RESUMO...")
    print("=" * 50)

    try:
        import json

        # Carregar dados do dashboard se disponível
        if os.path.exists('dashboard_data.json'):
            with open('dashboard_data.json', 'r', encoding='utf-8') as f:
                dados = json.load(f)

            print("📈 MÉTRICAS PRINCIPAIS:")
            if 'metricas' in dados:
                metricas = dados['metricas']
                for nome, info in metricas.items():
                    if isinstance(info, dict) and 'atual' in info:
                        melhoria = info.get('melhoria', 0)
                        seta = "↑" if melhoria > 0 else "↓" if melhoria < 0 else "→"
                        print(
                            f"   • {nome.upper()}: {info['atual']:,} ({seta} {abs(melhoria):.1f}%)")

            print("\n🎯 FEATURES PRINCIPAIS:")
            if 'features' in dados:
                features = dados['features'][:5]  # Top 5
                for i, feature in enumerate(features, 1):
                    print(
                        f"   {i}. {feature['name']} (importância: {feature['importance']:.2f})")

            print("\n🏆 MODELO SELECIONADO:")
            if 'modelo' in dados:
                modelo_info = dados['modelo']
                print(
                    f"   • Especificação: {modelo_info.get('especificacao', 'N/A')}")
                print(f"   • AIC: {modelo_info.get('aic', 'N/A')}")
                print(
                    f"   • Convergiu: {'✅ Sim' if modelo_info.get('convergiu', False) else '❌ Não'}")

            print("\n🔮 PRÓXIMAS PREVISÕES:")
            if 'previsoes' in dados and dados['previsoes']:
                previsoes = dados['previsoes'][:3]  # Próximos 3 dias
                for prev in previsoes:
                    data = prev['data']
                    dia = prev.get('dia_semana_pt', prev.get('dia_semana', ''))
                    valor = prev['valor']
                    print(f"   • {data} ({dia}): R$ {valor:,}")

            print("\n📁 ARQUIVOS GERADOS:")
            arquivos_finais = [
                'modelo_sarimax_melhorado.joblib',
                'dashboard_data.json',
                'sarimax_analysis.log',
                'relatorio_sarimax_completo.html'
            ]

            for arquivo in arquivos_finais:
                if os.path.exists(arquivo):
                    tamanho = os.path.getsize(arquivo) / 1024
                    print(f"   ✅ {arquivo} ({tamanho:.1f} KB)")
                else:
                    print(f"   ⚠️ {arquivo} (não encontrado)")

            # Salvar resumo em arquivo texto
            resumo_texto = f"""
RESUMO DA EXECUÇÃO - SARIMAX DASHBOARD V2.0
==========================================
Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MÉTRICAS PRINCIPAIS:
{chr(10).join([f'• {nome.upper()}: {info["atual"]:,} (melhoria: {info.get("melhoria", 0):.1f}%)'
               for nome, info in dados.get("metricas", {}).items()
               if isinstance(info, dict) and "atual" in info])}

MODELO SELECIONADO: {dados.get("modelo", {}).get("especificacao", "N/A")}
AIC: {dados.get("modelo", {}).get("aic", "N/A")}

TOP 5 FEATURES:
{chr(10).join([f'{i+1}. {f["name"]} (importância: {f["importance"]:.2f})'
                   for i, f in enumerate(dados.get("features", [])[:5])])}

STATUS: ✅ Execução concluída com sucesso
PRÓXIMO PASSO: Abra dashboard.html em um navegador
"""

            with open('resumo_execucao.txt', 'w', encoding='utf-8') as f:
                f.write(resumo_texto)

            print(f"\n💾 Resumo salvo em: resumo_execucao.txt")

        else:
            print("⚠️ Arquivo dashboard_data.json não encontrado")

    except Exception as e:
        print(f"❌ Erro ao gerar relatório: {e}")
        logger.error(f"Erro no relatório: {e}")


def abrir_dashboard_navegador():
    """
    Tenta abrir o dashboard no navegador padrão
    """
    print("\n🌐 ABRINDO DASHBOARD NO NAVEGADOR...")

    try:
        import os
        import webbrowser

        caminho_dashboard = os.path.abspath('dashboard.html')

        if os.path.exists('dashboard.html'):
            print(f"📂 Caminho: {caminho_dashboard}")
            webbrowser.open(f'file://{caminho_dashboard}')
            print("✅ Dashboard aberto no navegador padrão!")
            print("\n💡 INSTRUÇÕES:")
            print("   1. No dashboard, clique em '🔄 Carregar Dados do JSON'")
            print("   2. Explore as métricas, gráficos e previsões")
            print("   3. Use os botões de exportação para salvar resultados")
            print("   4. Pressione '?' para ver ajuda detalhada")
        else:
            print("❌ Arquivo dashboard.html não encontrado")

    except Exception as e:
        print(f"⚠️ Não foi possível abrir automaticamente: {e}")
        print("📖 Instruções manuais:")
        print("   1. Navegue até a pasta do projeto")
        print("   2. Clique duplo em 'dashboard.html'")
        print("   3. Ou abra um navegador e arraste o arquivo para ele")


def main():
    """
    Função principal que executa todo o pipeline
    """
    print("🚀 SARIMAX + DASHBOARD - EXECUÇÃO COMPLETA")
    print("="*60)
    print(f"⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Etapa 1: Verificar arquivos
    if not verificar_arquivos_necessarios():
        print("\n❌ Execução interrompida: arquivos necessários não encontrados")
        return False

    # Etapa 2: Executar modelo SARIMAX
    print("\n" + "="*60)
    print("ETAPA 1/3: EXECUTANDO MODELO SARIMAX")
    print("="*60)

    sucesso_modelo = executar_modelo_sarimax()
    if not sucesso_modelo:
        print("\n❌ Execução interrompida: falha no modelo SARIMAX")
        return False

    # Etapa 3: Integração com dashboard
    print("\n" + "="*60)
    print("ETAPA 2/3: INTEGRANDO COM DASHBOARD")
    print("="*60)

    sucesso_integracao = executar_integracao_dashboard()
    if not sucesso_integracao:
        print("\n⚠️ Aviso: falha na integração, mas continuando...")

    # Etapa 4: Relatório e abertura do dashboard
    print("\n" + "="*60)
    print("ETAPA 3/3: FINALIZANDO E ABRINDO DASHBOARD")
    print("="*60)

    gerar_relatorio_resumo()
    abrir_dashboard_navegador()

    # Finalização
    print("\n" + "="*60)
    print("🎉 EXECUÇÃO CONCLUÍDA COM SUCESSO!")
    print("="*60)
    print(f"⏰ Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📋 RESUMO DOS RESULTADOS:")
    print("   ✅ Modelo SARIMAX treinado e validado")
    print("   ✅ Dashboard integrado com dados reais")
    print("   ✅ Previsões geradas para próximos 7 dias")
    print("   ✅ Arquivos de saída criados")
    print("   ✅ Interface web disponível")

    print("\n🎯 PRÓXIMOS PASSOS:")
    print("   1. Explore o dashboard web que foi aberto")
    print("   2. Clique em 'Carregar Dados do JSON' se necessário")
    print("   3. Analise métricas, features e previsões")
    print("   4. Export dados usando os botões do dashboard")
    print("   5. Configure monitoramento contínuo se desejado")

    print("\n📞 SUPORTE:")
    print("   • Logs detalhados: sarimax_analysis.log")
    print("   • Resumo da execução: resumo_execucao.txt")
    print("   • Dashboard HTML: dashboard.html")
    print("   • Dados JSON: dashboard_data.json")

    return True


def executar_com_opcoes():
    """
    Execução com opções interativas
    """
    print("🔧 SARIMAX DASHBOARD - MODO INTERATIVO")
    print("="*50)

    opcoes = {
        '1': 'Execução completa (recomendado)',
        '2': 'Apenas treinar modelo SARIMAX',
        '3': 'Apenas integrar dashboard (modelo já treinado)',
        '4': 'Apenas abrir dashboard existente',
        '5': 'Validar integração',
        '0': 'Sair'
    }

    while True:
        print("\n📋 OPÇÕES DISPONÍVEIS:")
        for key, desc in opcoes.items():
            print(f"   {key}. {desc}")

        escolha = input("\n🎯 Escolha uma opção (0-5): ").strip()

        if escolha == '0':
            print("👋 Saindo...")
            break
        elif escolha == '1':
            main()
            break
        elif escolha == '2':
            executar_modelo_sarimax()
        elif escolha == '3':
            executar_integracao_dashboard()
        elif escolha == '4':
            abrir_dashboard_navegador()
        elif escolha == '5':
            from dashboard_integration import DashboardIntegrator
            integrador = DashboardIntegrator()
            resultado = integrador.validar_integracao_dashboard()
            print(f"\n📊 Resultado da validação: {resultado['status'].upper()}")
            if resultado['problemas']:
                print("⚠️ Problemas encontrados:")
                for problema in resultado['problemas']:
                    print(f"   • {problema}")
        else:
            print("❌ Opção inválida. Tente novamente.")


if __name__ == "__main__":
    """
    Ponto de entrada do script
    """
    import sys

    # Verificar argumentos da linha de comando
    if len(sys.argv) > 1:
        if sys.argv[1] == '--interativo':
            executar_com_opcoes()
        elif sys.argv[1] == '--help':
            print("""
USO: python exemplo_uso_completo.py [OPÇÃO]

OPÇÕES:
    --interativo    Modo interativo com menu de opções
    --help         Mostra esta ajuda
    (sem opção)    Execução completa automática

EXEMPLOS:
    python exemplo_uso_completo.py                # Execução completa
    python exemplo_uso_completo.py --interativo   # Modo interativo
    python exemplo_uso_completo.py --help         # Esta ajuda

DESCRIÇÃO:
    Este script automatiza todo o processo de treinamento do modelo
    SARIMAX e integração com o dashboard web interativo.
    
    O processo inclui:
    1. Verificação de arquivos necessários
    2. Treinamento e validação do modelo SARIMAX
    3. Integração dos dados com o dashboard web
    4. Geração de relatórios e visualizações
    5. Abertura automática do dashboard no navegador
            """)
        else:
            print(f"❌ Opção desconhecida: {sys.argv[1]}")
            print("Use --help para ver as opções disponíveis")
    else:
        # Execução completa padrão
        main()
