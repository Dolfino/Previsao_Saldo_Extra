#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verificador Final dos Dados do Dashboard
=======================================

Confirma se TODAS as informações estão sendo extraídas corretamente
do dashboard_data.json e prontas para alimentar o dashboard.
"""

import json
import os
from datetime import datetime


def verificar_dados_dashboard():
    """Verifica detalhadamente os dados do dashboard"""
    print("🔍 VERIFICAÇÃO FINAL DOS DADOS DO DASHBOARD")
    print("=" * 60)

    if not os.path.exists('dashboard_data.json'):
        print("❌ dashboard_data.json não encontrado!")
        return False

    try:
        with open('dashboard_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        print("✅ JSON carregado com sucesso\n")

        # 1. VERIFICAR TIMESTAMP
        print("⏰ TIMESTAMP E METADATA:")
        print("-" * 40)
        if 'timestamp' in data:
            timestamp = data['timestamp']
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                print(f"   📅 Gerado em: {dt.strftime('%d/%m/%Y %H:%M:%S')}")
                idade = datetime.now() - dt.replace(tzinfo=None)
                print(f"   ⏱️ Idade: {idade}")
            except:
                print(f"   📅 Timestamp: {timestamp}")

        # 2. VERIFICAR MÉTRICAS DETALHADAMENTE
        print("\n📊 MÉTRICAS DE PERFORMANCE:")
        print("-" * 40)
        if 'metricas' in data:
            metricas = data['metricas']
            for nome, valores in metricas.items():
                if isinstance(valores, dict):
                    atual = valores.get('atual', 'N/A')
                    original = valores.get('original', 'N/A')
                    melhoria = valores.get('melhoria', 'N/A')

                    # Formatação específica por métrica
                    if nome in ['rmse', 'mae']:
                        atual_fmt = f"{atual:,.0f}" if isinstance(
                            atual, (int, float)) else atual
                        original_fmt = f"{original:,.0f}" if isinstance(
                            original, (int, float)) else original
                    else:
                        atual_fmt = f"{atual:.3f}" if isinstance(
                            atual, (int, float)) else atual
                        original_fmt = f"{original:.3f}" if isinstance(
                            original, (int, float)) else original

                    print(f"   • {nome.upper()}:")
                    print(f"     Atual: {atual_fmt}")
                    print(f"     Original: {original_fmt}")
                    print(f"     Melhoria: {melhoria}%")
                    print()
        else:
            print("   ❌ Seção 'metricas' não encontrada")

        # 3. VERIFICAR FEATURES DETALHADAMENTE
        print("\n🎯 FEATURES SELECIONADAS:")
        print("-" * 40)
        if 'features' in data and isinstance(data['features'], list):
            features = data['features']
            print(f"   📋 Total: {len(features)} features")
            print("\n   🏆 Detalhamento:")

            for i, feature in enumerate(features, 1):
                if isinstance(feature, dict):
                    nome = feature.get('name', 'N/A')
                    tipo = feature.get('type', 'N/A')
                    imp = feature.get('importance', 'N/A')
                    desc = feature.get('description', 'Sem descrição')

                    print(f"      {i:2d}. {nome}")
                    print(f"          Tipo: {tipo}")
                    print(f"          Importância: {imp}")
                    print(
                        f"          Descrição: {desc[:60]}{'...' if len(desc) > 60 else ''}")
                    print()
        else:
            print("   ❌ Features não encontradas ou formato inválido")

        # 4. VERIFICAR INFORMAÇÕES DO MODELO
        print("\n🏗️ INFORMAÇÕES DO MODELO:")
        print("-" * 40)
        if 'modelo' in data:
            modelo = data['modelo']
            spec = modelo.get('especificacao', 'N/A')
            aic = modelo.get('aic', 'N/A')
            bic = modelo.get('bic', 'N/A')
            params = modelo.get('parametros', 'N/A')
            convergiu = modelo.get('convergiu', 'N/A')

            print(f"   📋 Especificação: {spec}")
            print(f"   📊 AIC: {aic}")
            print(f"   📊 BIC: {bic}")
            print(f"   🔧 Parâmetros: {params}")
            print(f"   ✅ Convergiu: {convergiu}")
        else:
            print("   ❌ Informações do modelo não encontradas")

        # 5. VERIFICAR CONFIGURAÇÃO
        print("\n⚙️ CONFIGURAÇÃO DO DATASET:")
        print("-" * 40)
        if 'configuracao' in data:
            config = data['configuracao']
            inicio = config.get('periodo_inicio', 'N/A')
            fim = config.get('periodo_fim', 'N/A')
            obs = config.get('total_observacoes', 'N/A')
            target = config.get('target_col', 'N/A')
            exog = config.get('exog_cols', [])

            print(f"   📅 Período: {inicio} até {fim}")
            print(f"   📊 Observações: {obs}")
            print(f"   🎯 Target: {target}")
            print(f"   📋 Exógenas: {exog}")
        else:
            print("   ❌ Configuração não encontrada")

        # 6. VERIFICAR DADOS OPCIONAIS
        secoes_opcionais = {
            'serie_temporal': 'Dados históricos',
            'previsoes': 'Previsões futuras',
            'outliers': 'Outliers detectados',
            'comparacao_modelos': 'Comparação de modelos'
        }

        print("\n📋 DADOS OPCIONAIS:")
        print("-" * 40)
        for secao, descricao in secoes_opcionais.items():
            if secao in data:
                dados_secao = data[secao]
                if isinstance(dados_secao, list):
                    print(f"   ✅ {descricao}: {len(dados_secao)} itens")
                elif isinstance(dados_secao, dict):
                    print(f"   ✅ {descricao}: {len(dados_secao)} elementos")
                else:
                    print(f"   ✅ {descricao}: Presente")
            else:
                print(f"   ⚠️ {descricao}: Ausente")

        # 7. VERIFICAR COMPATIBILIDADE COM DASHBOARD
        print("\n🌐 COMPATIBILIDADE COM DASHBOARD:")
        print("-" * 40)

        checks_dashboard = []

        # Check 1: Métricas têm formato correto
        if 'metricas' in data:
            metricas_ok = all(
                isinstance(data['metricas'].get(m, {}),
                           dict) and 'atual' in data['metricas'].get(m, {})
                for m in ['rmse', 'mae', 'r2', 'mape']
            )
            checks_dashboard.append(("Métricas formatadas", metricas_ok))

        # Check 2: Features têm estrutura correta
        if 'features' in data and isinstance(data['features'], list):
            features_ok = all(
                isinstance(f, dict) and 'name' in f and 'importance' in f
                for f in data['features']
            )
            checks_dashboard.append(("Features estruturadas", features_ok))

        # Check 3: Modelo tem informações básicas
        if 'modelo' in data:
            modelo_ok = 'especificacao' in data['modelo'] and 'aic' in data['modelo']
            checks_dashboard.append(("Modelo informado", modelo_ok))

        # Check 4: Timestamp válido
        timestamp_ok = 'timestamp' in data
        checks_dashboard.append(("Timestamp presente", timestamp_ok))

        # Mostrar resultados dos checks
        for check_nome, check_ok in checks_dashboard:
            status = "✅" if check_ok else "❌"
            print(f"   {status} {check_nome}")

        # Score final
        checks_ok = sum(1 for _, ok in checks_dashboard if ok)
        total_checks = len(checks_dashboard)
        score = (checks_ok / total_checks) * 100

        print(f"\n🎯 SCORE DE COMPATIBILIDADE: {score:.0f}%")
        print(f"✅ Checks OK: {checks_ok}/{total_checks}")

        # 8. SIMULAÇÃO DE CARREGAMENTO NO DASHBOARD
        print("\n🔮 SIMULAÇÃO DE CARREGAMENTO NO DASHBOARD:")
        print("-" * 40)

        # Simular o que o JavaScript do dashboard faria
        print("   📊 Dados que aparecerão no dashboard:")

        if 'metricas' in data and 'rmse' in data['metricas']:
            rmse = data['metricas']['rmse'].get('atual', 'N/A')
            melhoria = data['metricas']['rmse'].get('melhoria', 'N/A')
            print(f"      RMSE: {rmse:,.0f} ({melhoria:+.1f}% vs anterior)")

        if 'features' in data and len(data['features']) > 0:
            top_feature = data['features'][0]
            nome = top_feature.get('name', 'N/A')
            imp = top_feature.get('importance', 'N/A')
            print(f"      Top Feature: {nome} (importância: {imp})")

        if 'modelo' in data:
            spec = data['modelo'].get('especificacao', 'N/A')
            print(f"      Modelo: {spec}")

        print("\n🎉 VERIFICAÇÃO CONCLUÍDA!")

        if score >= 100:
            print("✅ PERFEITO! Todos os dados estão prontos para o dashboard")
            print("🚀 Pode abrir o dashboard e carregar os dados com confiança")
        elif score >= 75:
            print("✅ MUITO BOM! Dados estão em boa forma")
            print("⚠️ Algumas funcionalidades podem ter limitações")
        else:
            print("⚠️ ATENÇÃO! Alguns dados podem estar incompletos")
            print("🔧 Recomenda-se executar novamente o modelo")

        return score >= 75

    except json.JSONDecodeError as e:
        print(f"❌ Erro ao ler JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro na verificação: {e}")
        return False


def main():
    """Função principal"""
    print("🎯 VERIFICADOR FINAL - DADOS DO DASHBOARD")
    print("=" * 50)
    print("Confirmação detalhada de que todos os dados estão")
    print("corretamente estruturados para alimentar o dashboard.\n")

    sucesso = verificar_dados_dashboard()

    if sucesso:
        print("\n" + "="*60)
        print("🎊 DADOS VALIDADOS COM SUCESSO!")
        print("🌐 Agora abra dashboard.html e clique em 'Carregar Dados do JSON'")
        print("📊 Todos os dados reais devem aparecer corretamente!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("⚠️ PROBLEMAS DETECTADOS NOS DADOS")
        print("🔧 Execute novamente o modelo SARIMAX")
        print("="*60)

    return sucesso


if __name__ == "__main__":
    resultado = main()
    exit(0 if resultado else 1)
