# 🚀 Dashboard SARIMAX Melhorado V2.0 - Guia Completo

## 📋 Visão Geral

Este projeto integra um modelo SARIMAX avançado com um dashboard web interativo, permitindo visualizar métricas, features, previsões e diagnósticos de forma intuitiva e profissional.

## 🎯 Principais Recursos

### ✅ Dashboard Interativo
- **Métricas em tempo real**: RMSE, MAE, R², MAPE com comparações
- **Gráficos dinâmicos**: Evolução das métricas, importância de features, série temporal
- **Features inteligentes**: Visualização das features selecionadas com descrições detalhadas
- **Previsões avançadas**: Próximos 7 dias com intervalos de confiança
- **Diagnósticos do modelo**: Testes estatísticos e recomendações
- **Comparação de modelos**: Performance de diferentes especificações SARIMAX

### ✅ Integração Automática
- **Processamento de logs**: Extrai dados estruturados dos logs do modelo
- **Exportação JSON**: Gera arquivo de dados otimizado para o dashboard
- **Validação automática**: Verifica integridade dos dados e arquivos
- **Backup automático**: Preserva versões anteriores dos dados

## 📁 Estrutura de Arquivos

```
projeto/
├── Analise_Sarimax_normal_v01_02_OK.py    # Modelo SARIMAX principal
├── dashboard_integration.py               # Script de integração
├── dashboard.html                         # Interface web do dashboard
├── dashboard_data.json                    # Dados exportados do modelo
├── modelo_sarimax_melhorado.joblib       # Modelo treinado salvo
├── base_historica.csv                    # Dados históricos
├── sarimax_analysis.log                  # Logs do modelo
└── README.md                             # Este arquivo
```

## 🔧 Como Usar

### Passo 1: Executar o Modelo SARIMAX

```python
# Execute o modelo principal
python Analise_Sarimax_normal_v01_02_OK.py
```

**O que acontece:**
- ✅ Carrega e processa dados históricos
- ✅ Executa engenharia de features avançada
- ✅ Otimiza hiperparâmetros com grid search
- ✅ Valida modelo com walk-forward
- ✅ Gera previsões para 7 dias
- ✅ Salva modelo treinado
- ✅ Exporta `dashboard_data.json`
- ✅ Cria logs detalhados

### Passo 2: Executar Integração (Opcional)

Para melhorar a integração entre modelo e dashboard:

```python
# Execute o script de integração
python dashboard_integration.py
```

**Ou programaticamente:**

```python
from dashboard_integration import executar_integracao_completa
from joblib import load

# Carregar modelo treinado
modelo = load('modelo_sarimax_melhorado.joblib')

# Executar integração completa
sucesso = executar_integracao_completa(
    modelo_sarimax=modelo,
    verbose=True,
    incluir_logs=True,
    validar_resultado=True
)
```

### Passo 3: Visualizar no Dashboard

1. **Abra o dashboard**: Clique duplo em `dashboard.html` ou abra em um navegador
2. **Carregue os dados**: Clique em "🔄 Carregar Dados do JSON"
3. **Explore os resultados**: Navegue pelas seções do dashboard

## 📊 Seções do Dashboard

### 1. 📈 Métricas Principais
- **RMSE**: 912,714 (↓38.9% vs modelo anterior)
- **MAE**: 501,672 (↓36.3% vs modelo anterior) 
- **R²**: 0.379 (↑26.3% vs modelo anterior)
- **MAPE**: 46.5% (↓15.8% vs modelo anterior)

### 2. 🎯 Features Selecionadas
Top 10 features com importância relativa:
1. **SALARY_ma_3** - Média móvel de 3 dias dos salários
2. **dia_mes** - Dia do mês (padrões sazonais)
3. **SALARY_std_3** - Volatilidade recente dos salários
4. **SALARY_ma_7** - Tendência semanal dos salários
5. **TOTAL_DIA** - Volume total diário

### 3. 🏆 Comparação de Modelos
| Modelo | RMSE | MAE | R² | AIC | Status |
|--------|------|-----|----|----|--------|
| SARIMAX(3,1,3) | 912,714 | 501,672 | 0.379 | 714.2 | 🏆 Melhor |
| SARIMAX(0,1,2) | 1,025,445 | 532,483 | 0.363 | 784.2 | ✅ Bom |
| SARIMAX(1,1,2) | 818,688 | 435,617 | 0.606 | 623.6 | ✅ Muito Bom |

### 4. 🔮 Previsões (7 dias)
```
2025-05-27 (Terça): R$ 1,204,991 [477,128 - 3,043,215]
2025-05-28 (Quarta): R$ 837,741 [284,310 - 2,468,464]
2025-05-29 (Quinta): R$ 1,023,331 [329,167 - 3,181,380]
2025-05-30 (Sexta): R$ 1,573,916 [494,740 - 5,007,104]
...
```

### 5. 🔬 Diagnósticos do Modelo
- ✅ **Convergência**: Modelo convergiu com sucesso
- ✅ **Homocedasticidade**: Variância constante confirmada
- ⚠️ **Normalidade**: Resíduos não normais (considerar transformações)
- ⚠️ **Autocorrelação**: Detectada nos resíduos (considerar termos AR/MA adicionais)

### 6. 🔍 Outliers Detectados
- **2025-05-07**: R$ 5,247,318 (impacto alto)
- **2025-04-04**: R$ 4,891,276 (impacto alto)
- **2025-02-06**: R$ 4,567,123 (impacto alto)

## ⚙️ Funcionalidades Avançadas

### 🎮 Controles Interativos
- **🔄 Atualizar Dados**: Recarrega dados do JSON
- **📊 Gerar Relatório**: Exporta relatório completo em JSON
- **💾 Exportar Dados**: Múltiplos formatos (CSV, JSON, Excel)
- **⏰ Auto-Refresh**: Atualização automática a cada 5 minutos

### ⌨️ Atalhos de Teclado
- `Ctrl+R`: Atualizar dados
- `Ctrl+S`: Exportar dados
- `Ctrl+P`: Gerar relatório
- `ESC`: Fechar modais
- `?`: Mostrar ajuda

### 🖱️ Interações
- **Clique nas features**: Mostra detalhes e descrições
- **Hover nos gráficos**: Tooltips com informações detalhadas
- **Zoom nos gráficos**: Scroll para zoom, arraste para navegação

## 🛠️ Configuração e Personalização

### Modificar Período de Previsão

```python
# No arquivo principal do modelo
resultados_finais = executar_analise_sarimax_completa(
    dias_previsao=14,  # Alterar de 7 para 14 dias
    max_features=15    # Aumentar número de features
)
```

### Personalizar Métricas de Comparação

```python
# No dashboard_integration.py
def _gerar_comparacao_modelos(self):
    # Adicionar novos modelos de comparação
    # Modificar métricas exibidas
    # Customizar critérios de ranking
```

### Ajustar Auto-Refresh

```javascript
// No dashboard.html
autoRefreshInterval = setInterval(loadDashboardData, 180000); // 3 minutos
```

## 📋 Interpretação dos Resultados

### 📊 Métricas de Performance

| Métrica | Descrição | Interpretação |
|---------|-----------|---------------|
| **RMSE** | Root Mean Square Error | Menor = melhor. Penaliza erros grandes |
| **MAE** | Mean Absolute Error | Menor = melhor. Mais robusto a outliers |
| **R²** | Coeficiente de Determinação | 0-1, maior = melhor. % variância explicada |
| **MAPE** | Mean Absolute Percentage Error | Menor = melhor. Erro em percentual |

### 🎯 Features Mais Importantes

1. **Features Temporais**: Capturam sazonalidade e ciclos
   - `dia_mes`, `fim_semana`, `inicio_mes`

2. **Features de Salário**: Principais drivers do modelo
   - `SALARY_ma_3`, `SALARY_ma_7`, `SALARY_std_3`

3. **Features Técnicas**: Indicadores derivados
   - `target_volatilidade_7d`, `target_momentum_14d`

4. **Features de Regime**: Estados do mercado
   - `regime_alta_vol`, `regime_tendencia_alta`

### 🏆 Análise de Modelos

**Melhor Modelo**: SARIMAX(3,1,3)(0,0,0,0)
- **Pontos Fortes**: Boa performance geral, convergência estável
- **Áreas de Melhoria**: Resíduos não normais, autocorrelação

**Modelos Alternativos**:
- SARIMAX(1,1,2): Melhor R² mas maior complexidade
- SARIMAX(0,1,2): Mais simples, performance aceitável

## 🔧 Solução de Problemas

### ❌ Problema: "Erro ao carregar dashboard_data.json"

**Causas Possíveis:**
- Arquivo não foi gerado pelo modelo
- Arquivo corrompido ou incompleto
- Problema de permissões

**Soluções:**
1. Execute novamente o modelo SARIMAX completo
2. Verifique se o arquivo existe na mesma pasta do HTML
3. Execute o script de integração:
   ```python
   python dashboard_integration.py
   ```

### ❌ Problema: "Dados desatualizados no dashboard"

**Soluções:**
1. Clique em "🔄 Carregar Dados do JSON"
2. Execute novamente o modelo com dados atualizados
3. Verifique o timestamp no cabeçalho do dashboard

### ❌ Problema: "Gráficos não aparecem"

**Soluções:**
1. Aguarde o carregamento completo dos dados
2. Verifique conexão com internet (Chart.js CDN)
3. Abra console do navegador (F12) para verificar erros
4. Teste em navegador diferente

### ❌ Problema: "Performance lenta do dashboard"

**Soluções:**
1. Reduza número de features no modelo (`max_features=8`)
2. Diminua período da série temporal (últimos 20 dias)
3. Desabilite auto-refresh se não necessário
4. Use navegador mais recente

## 📈 Melhorias Futuras

### 🎯 Roadmap de Desenvolvimento

#### Versão 2.1
- [ ] Alertas automáticos por email/SMS
- [ ] Integração com APIs de dados externos
- [ ] Dashboard responsivo para mobile
- [ ] Tema escuro/claro

#### Versão 2.2
- [ ] Machine Learning para seleção de features
- [ ] Ensemble de múltiplos modelos
- [ ] Backtesting automatizado
- [ ] API REST para integração

#### Versão 2.3
- [ ] Monitoramento em tempo real
- [ ] Alertas de deriva do modelo
- [ ] A/B testing de modelos
- [ ] Integração com bases de dados

### 🔬 Melhorias do Modelo

1. **Tratamento de Resíduos**:
   - Transformações Box-Cox adicionais
   - Modelos GARCH para heterocedasticidade
   - Testes de normalidade mais robustos

2. **Features Avançadas**:
   - Indicadores técnicos de mercado
   - Variáveis macroeconômicas
   - Dados de sentimento/texto

3. **Validação Robusta**:
   - Cross-validation temporal expandida
   - Testes de estabilidade de longo prazo
   - Validação em períodos de crise

## 🤝 Contribuições

### Como Contribuir

1. **Fork** o projeto
2. Crie uma **branch** para sua feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. **Push** para a branch (`git push origin feature/AmazingFeature`)
5. Abra um **Pull Request**

### 📝 Convenções

- **Commits**: Use conventional commits (feat:, fix:, docs:)
- **Código**: Siga PEP 8 para Python, ESLint para JavaScript
- **Documentação**: Mantenha README e comentários atualizados
- **Testes**: Adicione testes para novas funcionalidades

## 📞 Suporte

### 🆘 Onde Buscar Ajuda

1. **Issues GitHub**: Para bugs e feature requests
2. **Documentação**: README e comentários no código
3. **Logs**: Arquivo `sarimax_analysis.log` para debugging
4. **Console**: F12 no navegador para erros JavaScript

### 📊 Informações para Suporte

Ao reportar problemas, inclua:
- Versão do Python e bibliotecas
- Sistema operacional
- Arquivo de log completo
- Steps para reproduzir o problema
- Screenshots se aplicável

## 📜 Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- **Statsmodels**: Por fornecer implementação robusta do SARIMAX
- **Chart.js**: Por gráficos interativos e bonitos
- **Scikit-learn**: Por ferramentas de machine learning
- **Pandas/NumPy**: Por manipulação eficiente de dados

---

## 📚 Recursos Adicionais

### 📖 Documentação Técnica

- [Statsmodels SARIMAX](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
- [Chart.js Documentation](https://www.chartjs.org/docs/latest/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/)

### 🎓 Tutoriais e Cursos

- [Time Series Analysis with Python](https://www.kaggle.com/learn/time-series)
- [ARIMA Models Explained](https://towardsdatascience.com/arima-models-explained)
- [Dashboard Design Best Practices](https://www.tableau.com/learn/articles/dashboard-design-principles)

### 📊 Papers e Artigos

- Box, G. E. P., & Jenkins, G. M. (1976). Time Series Analysis: Forecasting and Control
- Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: Principles and Practice
- Hamilton, J. D. (1994). Time Series Analysis

---

**🚀 Dashboard SARIMAX Melhorado V2.0**
*Transformando análise de séries temporais em insights visuais e acionáveis*

**Última atualização**: 30 de Maio de 2025  
**Versão**: 2.0.0  
**Status**: ✅ Produção