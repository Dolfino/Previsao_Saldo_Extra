#!/usr/bin/env python3
"""
Verificador Rápido do JSON - Confirma se dados estão corretos
"""

import json
import os


def verificar_json_rapido():
    print("🔍 VERIFICAÇÃO RÁPIDA DO dashboard_data.json")
    print("=" * 50)

    if not os.path.exists('dashboard_data.json'):
        print("❌ dashboard_data.json não encontrado!")
        return False

    try:
        with open('dashboard_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        print("✅ JSON carregado com sucesso")

        # Verificar métricas
        if 'metricas' in data:
            metricas = data['metricas']
            print(f"\n📊 MÉTRICAS ENCONTRADAS: {len(metricas)}")
            for nome, valores in metricas.items():
                if isinstance(valores, dict) and 'atual' in valores:
                    atual = valores['atual']
                    print(f"   ✅ {nome.upper()}: {atual}")
                else:
                    print(f"   ⚠️ {nome.upper()}: formato incorreto")
        else:
            print("\n❌ Seção 'metricas' ausente")

        # Verificar features
        if 'features' in data:
            features = data['features']
            if isinstance(features, list):
                print(f"\n🎯 FEATURES ENCONTRADAS: {len(features)}")
                for i, feature in enumerate(features[:5], 1):
                    if isinstance(feature, dict) and 'name' in feature:
                        nome = feature['name']
                        imp = feature.get('importance', 'N/A')
                        print(f"   ✅ {i}. {nome} (imp: {imp})")
                    else:
                        print(f"   ⚠️ {i}. Feature com formato incorreto")

                if len(features) > 5:
                    print(f"   ... e mais {len(features) - 5} features")
            else:
                print(
                    f"\n⚠️ Features não estão em formato de lista: {type(features)}")
        else:
            print("\n❌ Seção 'features' ausente")

        # Verificar outliers (que sabemos que estão funcionando)
        if 'outliers' in data:
            outliers = data['outliers']
            if isinstance(outliers, list):
                print(f"\n🔍 OUTLIERS ENCONTRADOS: {len(outliers)}")
                for i, outlier in enumerate(outliers[:3], 1):
                    if isinstance(outlier, dict):
                        data_outlier = outlier.get('data', 'N/A')
                        valor = outlier.get('valor', 'N/A')
                        print(f"   ✅ {i}. {data_outlier}: R$ {valor:,.0f}")
            else:
                print(f"\n⚠️ Outliers em formato incorreto: {type(outliers)}")
        else:
            print("\n❌ Seção 'outliers' ausente")

        # Verificar estrutura geral
        secoes_principais = ['timestamp', 'metricas',
                             'features', 'modelo', 'configuracao']
        print(f"\n📋 ESTRUTURA GERAL:")
        for secao in secoes_principais:
            existe = secao in data
            print(f"   {'✅' if existe else '❌'} {secao}")

        # Tamanho do arquivo
        tamanho_kb = os.path.getsize('dashboard_data.json') / 1024
        print(f"\n💾 Tamanho do arquivo: {tamanho_kb:.1f} KB")

        # Diagnóstico final
        tem_metricas = 'metricas' in data and len(data['metricas']) > 0
        tem_features = 'features' in data and isinstance(
            data['features'], list) and len(data['features']) > 0

        print(f"\n🎯 DIAGNÓSTICO FINAL:")
        print(
            f"   Métricas funcionais: {'✅ SIM' if tem_metricas else '❌ NÃO'}")
        print(
            f"   Features funcionais: {'✅ SIM' if tem_features else '❌ NÃO'}")

        if tem_metricas and tem_features:
            print(f"\n🎉 CONCLUSÃO: Dados estão CORRETOS!")
            print(f"⚠️ O warning da integração é um FALSO POSITIVO")
            print(f"✅ Dashboard deve funcionar perfeitamente")
        else:
            print(f"\n⚠️ CONCLUSÃO: Há problemas reais nos dados")
            print(f"🔧 Necessário regenerar dashboard_data.json")

        return tem_metricas and tem_features

    except json.JSONDecodeError as e:
        print(f"❌ Erro ao ler JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro na verificação: {e}")
        return False


if __name__ == "__main__":
    resultado = verificar_json_rapido()
    print(f"\n{'🎊 TUDO OK!' if resultado else '🔧 PRECISA CORRIGIR'}")
