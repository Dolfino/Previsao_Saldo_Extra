#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extrator e Validador de Dados do Dashboard SARIMAX
================================================

Este script demonstra como extrair informações específicas que alimentam
o dashboard e valida se os dados estão sendo corretamente estruturados.

Uso:
    python extrator_dados_dashboard.py

Autor: Sistema SARIMAX Melhorado V2.0
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExtratorDadosDashboard:
    """
    Extrai e valida dados específicos que alimentam o dashboard
    """

    def __init__(self, json_path: str = 'dashboard_data.json'):
        self.json_path = json_path
        self.dados = None
        self.informacoes_extraidas = {}

    def carregar_dados(self) -> bool:
        """Carrega dados do JSON"""
        if not os.path.exists(self.json_path):
            logger.error(f"❌ Arquivo não encontrado: {self.json_path}")
            return False

        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.dados = json.load(f)
            logger.info(f"✅ Dados carregados de {self.json_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao carregar JSON: {e}")
            return False

    def extrair_metricas_performance(self) -> Dict:
        """Extrai métricas de performance do modelo"""
        logger.info("\n📊 EXTRAINDO MÉTRICAS DE PERFORMANCE")

        if not self.dados or 'metricas' not in self.dados:
            logger.warning("⚠️ Seção 'metricas' não encontrada")
            return {}

        metricas = self.dados['metricas']
        metricas_extraidas = {}

        # Lista de métricas esperadas
        metricas_esperadas = {
            'rmse': 'Root Mean Square Error',
            'mae': 'Mean Absolute Error',
            'r2': 'Coeficiente de Determinação',
            'mape': 'Mean Absolute Percentage Error'
        }

        for metrica_key, metrica_nome in metricas_esperadas.items():
            if metrica_key in metricas:
                dados_metrica = metricas[metrica_key]

                # Extrair informações estruturadas
                info_metrica = {
                    'nome': metrica_nome,
                    'valor_atual': dados_metrica.get('atual', 'N/A'),
                    'valor_original': dados_metrica.get('original', 'N/A'),
                    'melhoria_percentual': dados_metrica.get('melhoria', 'N/A'),
                    'desvio_padrao': dados_metrica.get('std', 'N/A')
                }

                metricas_extraidas[metrica_key] = info_metrica

                # Log formatado
                valor = info_metrica['valor_atual']
                melhoria = info_metrica['melhoria_percentual']

                if isinstance(valor, (int, float)):
                    valor_fmt = f"{valor:,.0f}" if metrica_key in [
                        'rmse', 'mae'] else f"{valor:.3f}"
                else:
                    valor_fmt = str(valor)

                status = "📈" if isinstance(
                    melhoria, (int, float)) and melhoria > 0 else "📉"
                logger.info(
                    f"   {status} {metrica_nome}: {valor_fmt} ({melhoria}% melhoria)")
            else:
                logger.warning(f"   ⚠️ {metrica_nome}: Não encontrada")

        self.informacoes_extraidas['metricas'] = metricas_extraidas
        return metricas_extraidas

    def extrair_features_selecionadas(self) -> List[Dict]:
        """Extrai informações das features selecionadas"""
        logger.info("\n🎯 EXTRAINDO FEATURES SELECIONADAS")

        if not self.dados or 'features' not in self.dados:
            logger.warning("⚠️ Seção 'features' não encontrada")
            return []

        features = self.dados['features']
        features_extraidas = []

        if not isinstance(features, list):
            logger.warning("⚠️ Features não estão em formato de lista")
            return []

        logger.info(f"   📋 Total de features: {len(features)}")

        # Categorizar features por tipo
        tipos_features = {}

        for i, feature in enumerate(features):
            if isinstance(feature, dict):
                nome = feature.get('name', f'Feature_{i+1}')
                tipo = feature.get('type', 'unknown')
                importancia = feature.get('importance', 0)
                descricao = feature.get('description', 'Sem descrição')

                # Estruturar informação
                info_feature = {
                    'rank': i + 1,
                    'nome': nome,
                    'tipo': tipo,
                    'importancia': importancia,
                    'descricao': descricao,
                    'categoria': self._categorizar_feature(nome, tipo)
                }

                features_extraidas.append(info_feature)

                # Agrupar por tipo
                if tipo not in tipos_features:
                    tipos_features[tipo] = []
                tipos_features[tipo].append(nome)
            else:
                logger.warning(f"   ⚠️ Feature {i+1}: Formato inválido")

        # Log resumido por categoria
        logger.info("   📊 Features por categoria:")
        for tipo, lista_features in tipos_features.items():
            logger.info(
                f"      • {tipo.upper()}: {len(lista_features)} features")

        # Top 5 features
        logger.info("   🏆 Top 5 features por importância:")
        top_features = sorted(features_extraidas, key=lambda x: x.get(
            'importancia', 0), reverse=True)[:5]
        for i, feature in enumerate(top_features, 1):
            nome = feature['nome']
            imp = feature['importancia']
            tipo = feature['tipo']
            logger.info(f"      {i}. {nome} (imp: {imp:.3f}, tipo: {tipo})")

        self.informacoes_extraidas['features'] = features_extraidas
        self.informacoes_extraidas['resumo_features'] = {
            'total': len(features_extraidas),
            'por_tipo': tipos_features,
            'top_5': top_features
        }

        return features_extraidas

    def extrair_informacoes_modelo(self) -> Dict:
        """Extrai informações do modelo SARIMAX"""
        logger.info("\n🏗️ EXTRAINDO INFORMAÇÕES DO MODELO")

        if not self.dados or 'modelo' not in self.dados:
            logger.warning("⚠️ Seção 'modelo' não encontrada")
            return {}

        modelo = self.dados['modelo']

        # Extrair informações estruturadas
        info_modelo = {
            'especificacao': modelo.get('especificacao', 'N/A'),
            'ordem_ar': self._extrair_ordem_ar(modelo.get('especificacao', '')),
            'ordem_ma': self._extrair_ordem_ma(modelo.get('especificacao', '')),
            'diferenciacao': self._extrair_diferenciacao(modelo.get('especificacao', '')),
            'aic': modelo.get('aic', 'N/A'),
            'bic': modelo.get('bic', 'N/A'),
            'log_likelihood': modelo.get('log_likelihood', 'N/A'),
            'n_parametros': modelo.get('parametros', 'N/A'),
            'convergiu': modelo.get('convergiu', 'N/A')
        }

        # Log formatado
        logger.info(f"   📋 Especificação: {info_modelo['especificacao']}")
        logger.info(f"   📊 AIC: {info_modelo['aic']}")
        logger.info(f"   📊 BIC: {info_modelo['bic']}")
        logger.info(f"   🔧 Parâmetros: {info_modelo['n_parametros']}")

        convergencia = "✅" if info_modelo['convergiu'] else "❌"
        logger.info(
            f"   {convergencia} Convergência: {info_modelo['convergiu']}")

        self.informacoes_extraidas['modelo'] = info_modelo
        return info_modelo

    def extrair_dados_serie_temporal(self) -> List[Dict]:
        """Extrai dados da série temporal"""
        logger.info("\n📈 EXTRAINDO DADOS DA SÉRIE TEMPORAL")

        if not self.dados or 'serie_temporal' not in self.dados:
            logger.warning("⚠️ Seção 'serie_temporal' não encontrada")
            return []

        serie = self.dados['serie_temporal']

        if not isinstance(serie, list):
            logger.warning("⚠️ Série temporal não está em formato de lista")
            return []

        # Processar dados da série
        dados_processados = []
        valores = []

        for i, ponto in enumerate(serie):
            if isinstance(ponto, dict):
                data = ponto.get('data', f'Data_{i+1}')
                valor = ponto.get('valor', 0)
                log_valor = ponto.get('log_valor', 0)

                info_ponto = {
                    'data': data,
                    'valor': valor,
                    'log_valor': log_valor,
                    'data_formatada': self._formatar_data(data)
                }

                dados_processados.append(info_ponto)
                if isinstance(valor, (int, float)):
                    valores.append(valor)

        # Estatísticas da série
        if valores:
            estatisticas = {
                'total_pontos': len(valores),
                'valor_medio': np.mean(valores),
                'valor_mediano': np.median(valores),
                'desvio_padrao': np.std(valores),
                'valor_minimo': np.min(valores),
                'valor_maximo': np.max(valores),
                'coef_variacao': np.std(valores) / np.mean(valores) if np.mean(valores) != 0 else 0
            }

            logger.info(
                f"   📊 Pontos na série: {estatisticas['total_pontos']}")
            logger.info(
                f"   💰 Valor médio: R$ {estatisticas['valor_medio']:,.0f}")
            logger.info(
                f"   📏 Range: R$ {estatisticas['valor_minimo']:,.0f} - R$ {estatisticas['valor_maximo']:,.0f}")
            logger.info(
                f"   📊 Coef. Variação: {estatisticas['coef_variacao']:.3f}")

            self.informacoes_extraidas['serie_temporal'] = {
                'dados': dados_processados,
                'estatisticas': estatisticas
            }
        else:
            logger.warning("   ⚠️ Nenhum valor numérico válido encontrado")

        return dados_processados

    def extrair_previsoes(self) -> List[Dict]:
        """Extrai dados das previsões"""
        logger.info("\n🔮 EXTRAINDO PREVISÕES")

        if not self.dados or 'previsoes' not in self.dados:
            logger.warning("⚠️ Seção 'previsoes' não encontrada")
            return []

        previsoes = self.dados['previsoes']

        if not isinstance(previsoes, list):
            logger.warning("⚠️ Previsões não estão em formato de lista")
            return []

        previsoes_processadas = []
        valores_previstos = []

        for i, previsao in enumerate(previsoes):
            if isinstance(previsao, dict):
                data = previsao.get('data', f'Data_{i+1}')
                valor = previsao.get('valor', 0)
                dia_semana = previsao.get('dia_semana', 'N/A')
                intervalo_inf = previsao.get('intervalo_inferior', None)
                intervalo_sup = previsao.get('intervalo_superior', None)

                info_previsao = {
                    'data': data,
                    'valor': valor,
                    'dia_semana': dia_semana,
                    'intervalo_inferior': intervalo_inf,
                    'intervalo_superior': intervalo_sup,
                    'tem_intervalo': intervalo_inf is not None and intervalo_sup is not None,
                    'largura_intervalo': (intervalo_sup - intervalo_inf) if (intervalo_inf and intervalo_sup) else None
                }

                previsoes_processadas.append(info_previsao)
                if isinstance(valor, (int, float)):
                    valores_previstos.append(valor)

        # Estatísticas das previsões
        if valores_previstos:
            stats_previsoes = {
                'total_previsoes': len(valores_previstos),
                'valor_medio_previsto': np.mean(valores_previstos),
                'desvio_previsoes': np.std(valores_previstos),
                'tendencia': 'crescente' if valores_previstos[-1] > valores_previstos[0] else 'decrescente'
            }

            logger.info(
                f"   📊 Total de previsões: {stats_previsoes['total_previsoes']}")
            logger.info(
                f"   💰 Valor médio previsto: R$ {stats_previsoes['valor_medio_previsto']:,.0f}")
            logger.info(f"   📈 Tendência: {stats_previsoes['tendencia']}")

            # Mostrar previsões
            logger.info("   📅 Previsões detalhadas:")
            for prev in previsoes_processadas[:5]:  # Mostrar primeiras 5
                data = prev['data']
                valor = prev['valor']
                dia = prev['dia_semana']
                logger.info(f"      • {data} ({dia}): R$ {valor:,.0f}")

            self.informacoes_extraidas['previsoes'] = {
                'dados': previsoes_processadas,
                'estatisticas': stats_previsoes
            }

        return previsoes_processadas

    def extrair_outliers(self) -> List[Dict]:
        """Extrai informações dos outliers"""
        logger.info("\n🔍 EXTRAINDO OUTLIERS")

        if not self.dados or 'outliers' not in self.dados:
            logger.warning("⚠️ Seção 'outliers' não encontrada")
            return []

        outliers = self.dados['outliers']

        if not isinstance(outliers, list):
            logger.warning("⚠️ Outliers não estão em formato de lista")
            return []

        outliers_processados = []
        valores_outliers = []

        for outlier in outliers:
            if isinstance(outlier, dict):
                data = outlier.get('data', 'N/A')
                valor = outlier.get('valor', 0)
                impacto = outlier.get('impacto', 'N/A')

                info_outlier = {
                    'data': data,
                    'valor': valor,
                    'impacto': impacto,
                    'data_formatada': self._formatar_data(data)
                }

                outliers_processados.append(info_outlier)
                if isinstance(valor, (int, float)):
                    valores_outliers.append(valor)

        # Estatísticas dos outliers
        if valores_outliers:
            stats_outliers = {
                'total_outliers': len(valores_outliers),
                'valor_medio_outliers': np.mean(valores_outliers),
                'valor_max_outlier': np.max(valores_outliers),
                'valor_min_outlier': np.min(valores_outliers)
            }

            logger.info(
                f"   📊 Total de outliers: {stats_outliers['total_outliers']}")
            logger.info(
                f"   💰 Valor médio outliers: R$ {stats_outliers['valor_medio_outliers']:,.0f}")
            logger.info(
                f"   📈 Maior outlier: R$ {stats_outliers['valor_max_outlier']:,.0f}")

            # Mostrar top outliers
            outliers_ordenados = sorted(outliers_processados,
                                        key=lambda x: x.get('valor', 0), reverse=True)
            logger.info("   🎯 Top 3 outliers:")
            for i, outlier in enumerate(outliers_ordenados[:3], 1):
                data = outlier['data']
                valor = outlier['valor']
                impacto = outlier['impacto']
                logger.info(
                    f"      {i}. {data}: R$ {valor:,.0f} (impacto: {impacto})")

            self.informacoes_extraidas['outliers'] = {
                'dados': outliers_processados,
                'estatisticas': stats_outliers
            }

        return outliers_processados

    def extrair_configuracao(self) -> Dict:
        """Extrai informações de configuração"""
        logger.info("\n⚙️ EXTRAINDO CONFIGURAÇÃO")

        if not self.dados or 'configuracao' not in self.dados:
            logger.warning("⚠️ Seção 'configuracao' não encontrada")
            return {}

        config = self.dados['configuracao']

        info_config = {
            'periodo_inicio': config.get('periodo_inicio', 'N/A'),
            'periodo_fim': config.get('periodo_fim', 'N/A'),
            'total_observacoes': config.get('total_observacoes', 'N/A'),
            'features_criadas': config.get('total_features_criadas', 'N/A'),
            'features_selecionadas': config.get('features_selecionadas', 'N/A'),
            'target_col': config.get('target_col', 'N/A'),
            'exog_cols': config.get('exog_cols', [])
        }

        # Calcular duração se possível
        if (info_config['periodo_inicio'] != 'N/A' and
                info_config['periodo_fim'] != 'N/A'):
            try:
                inicio = datetime.strptime(
                    info_config['periodo_inicio'], '%Y-%m-%d')
                fim = datetime.strptime(info_config['periodo_fim'], '%Y-%m-%d')
                duracao = (fim - inicio).days
                info_config['duracao_dias'] = duracao
            except:
                info_config['duracao_dias'] = 'N/A'

        # Log formatado
        logger.info(
            f"   📅 Período: {info_config['periodo_inicio']} até {info_config['periodo_fim']}")
        logger.info(f"   📊 Observações: {info_config['total_observacoes']}")
        logger.info(f"   🎯 Target: {info_config['target_col']}")
        logger.info(f"   📋 Exógenas: {info_config['exog_cols']}")

        if info_config.get('duracao_dias', 'N/A') != 'N/A':
            logger.info(f"   ⏱️ Duração: {info_config['duracao_dias']} dias")

        self.informacoes_extraidas['configuracao'] = info_config
        return info_config

    def validar_completude_dados(self) -> Dict:
        """Valida a completude dos dados extraídos"""
        logger.info("\n✅ VALIDANDO COMPLETUDE DOS DADOS")

        validacao = {
            'secoes_obrigatorias': [],
            'secoes_opcionais': [],
            'problemas': [],
            'score_completude': 0
        }

        # Seções obrigatórias
        secoes_obrigatorias = {
            'metricas': 'Métricas de performance',
            'features': 'Features selecionadas',
            'modelo': 'Informações do modelo',
            'configuracao': 'Configuração geral'
        }

        # Seções opcionais
        secoes_opcionais = {
            'serie_temporal': 'Dados históricos',
            'previsoes': 'Previsões futuras',
            'outliers': 'Outliers detectados',
            'comparacao_modelos': 'Comparação de modelos'
        }

        total_secoes = len(secoes_obrigatorias) + len(secoes_opcionais)
        secoes_presentes = 0

        # Validar seções obrigatórias
        for secao, descricao in secoes_obrigatorias.items():
            if secao in self.informacoes_extraidas:
                validacao['secoes_obrigatorias'].append({
                    'secao': secao,
                    'descricao': descricao,
                    'presente': True
                })
                secoes_presentes += 1
                logger.info(f"   ✅ {descricao}: Presente")
            else:
                validacao['secoes_obrigatorias'].append({
                    'secao': secao,
                    'descricao': descricao,
                    'presente': False
                })
                validacao['problemas'].append(
                    f"Seção obrigatória ausente: {descricao}")
                logger.warning(f"   ❌ {descricao}: Ausente")

        # Validar seções opcionais
        for secao, descricao in secoes_opcionais.items():
            if secao in self.informacoes_extraidas:
                validacao['secoes_opcionais'].append({
                    'secao': secao,
                    'descricao': descricao,
                    'presente': True
                })
                secoes_presentes += 1
                logger.info(f"   ✅ {descricao}: Presente")
            else:
                validacao['secoes_opcionais'].append({
                    'secao': secao,
                    'descricao': descricao,
                    'presente': False
                })
                logger.info(f"   ℹ️ {descricao}: Opcional (ausente)")

        # Calcular score de completude
        validacao['score_completude'] = (secoes_presentes / total_secoes) * 100

        # Validações específicas de qualidade
        self._validar_qualidade_metricas(validacao)
        self._validar_qualidade_features(validacao)
        self._validar_qualidade_modelo(validacao)

        return validacao

    def _validar_qualidade_metricas(self, validacao: Dict) -> None:
        """Valida qualidade das métricas"""
        if 'metricas' not in self.informacoes_extraidas:
            return

        metricas = self.informacoes_extraidas['metricas']

        # Verificar se métricas têm valores válidos
        metricas_validas = 0
        total_metricas = len(metricas)

        for metrica, dados in metricas.items():
            valor_atual = dados.get('valor_atual', 'N/A')
            if isinstance(valor_atual, (int, float)) and valor_atual > 0:
                metricas_validas += 1
            else:
                validacao['problemas'].append(
                    f"Métrica {metrica} com valor inválido: {valor_atual}")

        if metricas_validas == total_metricas:
            logger.info(f"   ✅ Todas as {total_metricas} métricas são válidas")
        else:
            logger.warning(
                f"   ⚠️ Apenas {metricas_validas}/{total_metricas} métricas são válidas")

    def _validar_qualidade_features(self, validacao: Dict) -> None:
        """Valida qualidade das features"""
        if 'features' not in self.informacoes_extraidas:
            return

        features = self.informacoes_extraidas['features']

        if len(features) < 5:
            validacao['problemas'].append(
                f"Poucas features selecionadas: {len(features)}")
            logger.warning(
                f"   ⚠️ Apenas {len(features)} features (recomendado: ≥5)")
        else:
            logger.info(
                f"   ✅ {len(features)} features selecionadas (adequado)")

        # Verificar diversidade de tipos
        tipos_unicos = set()
        for feature in features:
            tipo = feature.get('tipo', 'unknown')
            tipos_unicos.add(tipo)

        if len(tipos_unicos) >= 3:
            logger.info(
                f"   ✅ Boa diversidade de tipos: {len(tipos_unicos)} tipos diferentes")
        else:
            logger.warning(
                f"   ⚠️ Pouca diversidade: apenas {len(tipos_unicos)} tipos de features")

    def _validar_qualidade_modelo(self, validacao: Dict) -> None:
        """Valida qualidade do modelo"""
        if 'modelo' not in self.informacoes_extraidas:
            return

        modelo = self.informacoes_extraidas['modelo']

        # Verificar se convergiu
        convergiu = modelo.get('convergiu', 'N/A')
        if convergiu == True:
            logger.info("   ✅ Modelo convergiu com sucesso")
        else:
            validacao['problemas'].append("Modelo não convergiu")
            logger.warning("   ❌ Modelo não convergiu")

        # Verificar AIC
        aic = modelo.get('aic', 'N/A')
        if isinstance(aic, (int, float)):
            logger.info(f"   ✅ AIC válido: {aic:.3f}")
        else:
            validacao['problemas'].append("AIC não disponível")
            logger.warning("   ⚠️ AIC não disponível")

    def gerar_relatorio_extracao(self) -> Dict:
        """Gera relatório completo da extração"""
        logger.info("\n📋 GERANDO RELATÓRIO DE EXTRAÇÃO")

        # Executar todas as extrações
        if not self.carregar_dados():
            return {}

        # Extrair todas as informações
        self.extrair_metricas_performance()
        self.extrair_features_selecionadas()
        self.extrair_informacoes_modelo()
        self.extrair_dados_serie_temporal()
        self.extrair_previsoes()
        self.extrair_outliers()
        self.extrair_configuracao()

        # Validar completude
        validacao = self.validar_completude_dados()

        # Estruturar relatório
        relatorio = {
            'timestamp_extracao': datetime.now().isoformat(),
            'arquivo_origem': self.json_path,
            'informacoes_extraidas': self.informacoes_extraidas,
            'validacao': validacao,
            'resumo': {
                'secoes_extraidas': len(self.informacoes_extraidas),
                'score_completude': validacao['score_completude'],
                'problemas_encontrados': len(validacao['problemas']),
                'status': 'completo' if validacao['score_completude'] > 80 else 'incompleto'
            }
        }

        # Log resumo final
        logger.info("\n" + "="*60)
        logger.info("📊 RESUMO DA EXTRAÇÃO")
        logger.info("="*60)

        logger.info(
            f"✅ Seções extraídas: {relatorio['resumo']['secoes_extraidas']}")
        logger.info(
            f"📊 Score de completude: {relatorio['resumo']['score_completude']:.1f}%")
        logger.info(
            f"⚠️ Problemas encontrados: {relatorio['resumo']['problemas_encontrados']}")
        logger.info(f"🎯 Status geral: {relatorio['resumo']['status'].upper()}")

        if validacao['problemas']:
            logger.info("\n⚠️ PROBLEMAS IDENTIFICADOS:")
            for i, problema in enumerate(validacao['problemas'], 1):
                logger.info(f"   {i}. {problema}")

        return relatorio

    def salvar_relatorio(self, relatorio: Dict, caminho: str = 'relatorio_extracao.json') -> bool:
        """Salva relatório em arquivo JSON"""
        try:
            with open(caminho, 'w', encoding='utf-8') as f:
                json.dump(relatorio, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Relatório salvo em: {caminho}")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao salvar relatório: {e}")
            return False

    # Métodos auxiliares
    def _categorizar_feature(self, nome: str, tipo: str) -> str:
        """Categoriza feature baseado no nome e tipo"""
        nome_lower = nome.lower()

        if 'salary' in nome_lower:
            return 'salario'
        elif any(temp in nome_lower for temp in ['dia', 'mes', 'semana']):
            return 'temporal'
        elif any(tech in nome_lower for tech in ['ma_', 'std_', 'lag_']):
            return 'tecnica'
        elif 'target' in nome_lower:
            return 'target_derivada'
        else:
            return tipo or 'outras'

    def _extrair_ordem_ar(self, especificacao: str) -> str:
        """Extrai ordem AR da especificação"""
        try:
            # SARIMAX(p,d,q)(P,D,Q,s) -> extrair p
            import re
            match = re.search(r'SARIMAX\((\d+),', especificacao)
            return match.group(1) if match else 'N/A'
        except:
            return 'N/A'

    def _extrair_ordem_ma(self, especificacao: str) -> str:
        """Extrai ordem MA da especificação"""
        try:
            # SARIMAX(p,d,q)(P,D,Q,s) -> extrair q
            import re
            match = re.search(r'SARIMAX\(\d+,\d+,(\d+)\)', especificacao)
            return match.group(1) if match else 'N/A'
        except:
            return 'N/A'

    def _extrair_diferenciacao(self, especificacao: str) -> str:
        """Extrai ordem de diferenciação da especificação"""
        try:
            # SARIMAX(p,d,q)(P,D,Q,s) -> extrair d
            import re
            match = re.search(r'SARIMAX\(\d+,(\d+),\d+\)', especificacao)
            return match.group(1) if match else 'N/A'
        except:
            return 'N/A'

    def _formatar_data(self, data_str: str) -> str:
        """Formata data para exibição"""
        try:
            # Tentar converter de ISO para formato brasileiro
            data = datetime.fromisoformat(data_str.replace('Z', '+00:00'))
            return data.strftime('%d/%m/%Y')
        except:
            # Se não conseguir, retornar como está
            return data_str


def main():
    """Função principal"""
    print("🔍 EXTRATOR E VALIDADOR DE DADOS DO DASHBOARD")
    print("="*55)
    print("Este script extrai e valida informações que alimentam o dashboard")
    print("para garantir que todos os dados estão sendo corretamente estruturados.\n")

    # Inicializar extrator
    extrator = ExtratorDadosDashboard()

    # Gerar relatório completo
    relatorio = extrator.gerar_relatorio_extracao()

    if relatorio:
        # Salvar relatório
        extrator.salvar_relatorio(relatorio)

        # Status final
        score = relatorio['resumo']['score_completude']
        problemas = relatorio['resumo']['problemas_encontrados']

        print("\n" + "="*60)
        if score > 90 and problemas == 0:
            print("🎉 EXTRAÇÃO PERFEITA!")
            print("✅ Todos os dados estão corretamente estruturados")
            print("🚀 Dashboard pode ser usado com confiança")
        elif score > 70:
            print("✅ EXTRAÇÃO BEM-SUCEDIDA!")
            print(f"📊 {score:.1f}% dos dados extraídos")
            if problemas > 0:
                print(f"⚠️ {problemas} problema(s) menor(es) identificado(s)")
        else:
            print("⚠️ EXTRAÇÃO INCOMPLETA")
            print(f"📊 Apenas {score:.1f}% dos dados extraídos")
            print(f"❌ {problemas} problema(s) encontrado(s)")
            print("🔧 Verifique o arquivo dashboard_data.json")

        print(f"\n📋 Relatório detalhado salvo em: relatorio_extracao.json")

        return score > 70
    else:
        print("❌ Falha na extração dos dados")
        print("🔧 Verifique se dashboard_data.json existe e é válido")
        return False


if __name__ == "__main__":
    sucesso = main()
    exit(0 if sucesso else 1)
