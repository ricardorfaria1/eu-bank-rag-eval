# Avaliação de RAG sobre regulamentação de AML (design)

Data: 2026-11-17
Estado: rascunho para aprovação

## Intenção

Segundo projeto de portfólio para entrevistas de engenheiro de IA. Mostra que o autor sabe construir um RAG e provar com números qual configuração é melhor, separando erros de busca de erros de resposta.

A equipa de compliance de um banco português fictício (o mesmo do projeto `eu-bank-mcp-server`) pergunta sobre legislação e orientações de combate ao branqueamento de capitais e recebe respostas que citam o documento e o artigo de origem. O projeto constrói esse RAG em três configurações, avalia as três com o mesmo conjunto de perguntas usando o Ragas, e publica um relatório comparativo.

Repositório independente. O terceiro projeto (teste A/B) poderá comparar duas destas configurações com rigor estatístico, mas não depende deste para correr.

## Referência e restrição de autoria

Inspira-se em `junidepieri-design/eval-001-rag-evaluation-template` e `genai-001-regulatory-rag-bsa-aml`. Esses repositórios não têm licença: nenhum código, texto ou diagrama pode ser copiado. Distingue-se por: corpus europeu e português, comparação de configurações em vez de avaliação única, conjunto de teste gerado e revisto com casos difíceis escritos à mão, e métrica própria de citação.

## Decisões do autor

- Público: equipa de compliance do banco.
- Domínio: legislação e orientações de AML da UE, do Banco de Portugal e da CMVM.
- Eixo da avaliação: comparar configurações do RAG.
- Conjunto de teste: gerado pelo Ragas, revisto pelo autor, mais casos difíceis escritos à mão.
- Corpus inclui CMVM e Lei 97/2017; o banco fictício tem uma gestora de fundos no grupo.
- Construção: pacote Python com um comando por experiência e relatório em Markdown versionado.
- Stack: LangChain, Qdrant, divisão por artigo, embeddings multilingues, Claude.
- Código em inglês; README e documentação em inglês e português.

## Objetivos

- Pipeline de ingestão que transforma os documentos oficiais em trechos com metadados (documento, artigo, língua, fonte).
- Três configurações do RAG, cada uma a mudar uma só coisa em relação à anterior.
- Respostas que citam documento e artigo, na língua da pergunta, e que recusam quando os trechos não chegam.
- Conjunto de teste de cerca de 50 perguntas com referências revistas.
- Avaliação com Ragas que separa busca de resposta, mais citação, recusa, custo e tempo.
- Relatório comparativo em Markdown, com tabela e gráfico, versionado no repositório.
- Testes unitários sem rede nem API, a correr no CI.

## Não objetivos

- Interface de chat ou servidor web.
- Avaliação contínua em produção ou portão de qualidade no CI.
- Painel de experiências (MLflow, Langfuse). Fica como próximo passo no README.
- Afinação de modelos de embeddings ou de LLM.
- Aconselhamento jurídico: o README diz que as respostas não substituem o texto oficial.

## Anti-objetivos

O projeto falha, mesmo a funcionar, se:

- parecer uma cópia dos repositórios de referência;
- os números assentarem em referências que ninguém reviu;
- o relatório declarar um vencedor sem mostrar custo, tempo e a dimensão da diferença;
- as respostas não disserem de onde vem a informação;
- o autor não conseguir explicar numa entrevista porque uma configuração ganhou.

## Restrições

- Python, uv, LangChain, Qdrant, Ragas, Docling, Claude via SDK da Anthropic (ou o integrador do LangChain para a Anthropic).
- Corre num portátil macOS sem GPU.
- Os documentos não vão para o repositório: um script descarrega-os das fontes oficiais. O repositório guarda o manifesto (URL, data de consulta, hash).
- Chamadas pagas (geração do conjunto de teste, respostas, juiz) só correm com aprovação do autor e com estimativa de custo mostrada antes.
- Orçamento (decisão do autor): respostas no Claude Opus 5.5, juiz do Ragas no Claude Haiku 5.5. Cerca de 8 USD por execução completa das três configurações e 3 a 10 USD para gerar o conjunto de teste; teto total de 25 USD salvo nova aprovação.
- A medição da busca não usa LLM: compara os trechos recuperados com o documento e o artigo esperados de cada pergunta, e corre sem custo.

## Abordagens consideradas

| Abordagem | Troca | Decisão |
|---|---|---|
| A. Pacote Python, configurações em YAML, um comando por experiência, relatório Markdown versionado | Mais estrutura que um notebook; trocar configuração é editar um ficheiro | **Escolhida pelo autor** |
| B. Notebook Jupyter | Rápido e legível no GitHub, mas mistura código, configuração e resultados, difícil de testar e próximo da referência | Rejeitada |
| C. Pacote com rastreio de experiências (MLflow ou Langfuse) | Mais próximo de produção, mas acrescenta um serviço; o diferencial é a comparação | Rejeitada; próximo passo |

Eixo da avaliação também foi escolhido entre três (comparar configurações, portão no CI, relatório único). O autor escolheu comparar configurações.

## Design

### Corpus

| Origem | Documento | Língua |
|---|---|---|
| UE | Regulamento (UE) 2024/1624 (AMLR) | EN e PT |
| UE | Diretiva (UE) 2024/1640 (AMLD6) | EN e PT |
| UE | Regulamento (UE) 2024/1620 (AMLA) | EN e PT |
| EBA | Orientações sobre fatores de risco de BC/FT | EN |
| EBA | Orientações sobre onboarding remoto de clientes | EN |
| PT | Lei 83/2017 | PT |
| PT | Lei 97/2017 (medidas restritivas) | PT |
| BdP | Aviso do Banco de Portugal n.º 1/2022 | PT |
| BdP | Instrução do Banco de Portugal n.º 8/2024 | PT |
| CMVM | Regulamento da CMVM n.º 2/2020, versão consolidada | PT |

Os atos da UE entram numa só língua por configuração base, para o índice não ter duplicados: inglês. A versão portuguesa serve para casos de teste que comparam línguas (decisão do construtor do design; o construtor pode mudar se encontrar razão).

Primeira tarefa da construção: confirmar no site do Banco de Portugal e da CMVM que estas são as normas em vigor e se falta alguma relevante. O autor quer o conjunto do Banco de Portugal completo: todos os Avisos e Instruções em vigor sobre BC/FT, mais cartas circulares e orientações publicadas especificamente sobre BC/FT; FAQ ficam de fora. O construtor pode trocar ou acrescentar documentos, mas mostra a lista final ao autor antes de gerar o conjunto de teste, porque ela muda o custo e os casos difíceis. (Decisão do construtor do design.)

Atos da UE: preferir HTML do EUR-Lex, que preserva a estrutura de artigos. PDFs (EBA, BdP, CMVM): Docling. Leis portuguesas: HTML ou PDF do Diário da República, à escolha do construtor.

### Divisão em trechos

- **Por tamanho fixo:** cerca de 1000 caracteres com 150 de sobreposição, através do divisor recursivo do LangChain.
- **Por artigo:** um trecho por artigo; artigos longos divididos por número ou alínea, mantendo o cabeçalho do artigo em cada parte.

Metadados de cada trecho: id do documento, título curto, número do artigo (quando existe), língua, URL de origem, data de consulta.

### Configurações

| Id | Divisão | Busca |
|---|---|---|
| `baseline` | tamanho fixo | só semântica |
| `by-article` | por artigo | só semântica |
| `by-article-hybrid` | por artigo | híbrida (semântica + palavras) |

Cada configuração é um ficheiro YAML com: divisão, tipo de busca, número de trechos devolvidos (5 por omissão), modelo de embeddings, modelo de geração. Acrescentar uma quarta configuração é acrescentar um ficheiro.

Embeddings: BGE-M3, local. Na busca híbrida, a parte de palavras pode usar os vetores esparsos do próprio BGE-M3 ou BM25; escolha do construtor. Qdrant em modo local (sem Docker), desde que suporte vetores esparsos; caso contrário, Docker.

### Geração da resposta

Claude, com instruções para:

- responder só com base nos trechos recebidos;
- citar cada afirmação no formato `[documento, art. N]`;
- responder na língua da pergunta;
- dizer que não encontrou a resposta quando os trechos não chegam, em vez de adivinhar.

### Conjunto de teste

Ficheiro CSV versionado com as colunas: `id`, `language`, `user_input`, `reference`, `reference_contexts`, `reference_citations` (documento e artigo esperados), `category`, `origin` (`generated`, `reviewed`, `handwritten`), `review_note`.

1. O gerador de testes do Ragas cria cerca de 80 pares a partir do corpus.
2. O autor revê numa folha de cálculo: aprova, corrige ou rejeita, e anota o que mudou. Ficam cerca de 40.
3. O autor e o Claude escrevem cerca de 10 casos difíceis:
   - perguntas que precisam de dois documentos (por exemplo, AMLR e Aviso 1/2022);
   - pergunta em português sobre um texto que está no índice em inglês;
   - norma revogada ("o Aviso 2/2018 ainda está em vigor?");
   - obrigações que mudam entre supervisores (Banco de Portugal e CMVM);
   - perguntas fora do assunto, que devem ser recusadas.

Só linhas com `origin` em `reviewed` ou `handwritten` entram na avaliação.

Línguas: os utilizadores são uma equipa portuguesa, por isso cerca de 60% das perguntas ficam em português e 40% em inglês. O gerador é orientado para isso: gera a partir dos documentos portugueses, e parte das perguntas sobre atos da UE é traduzida para português antes da revisão. A coluna `language` regista a língua de cada pergunta, e o relatório mostra os resultados por língua, porque a busca por palavras deve sofrer com perguntas em português sobre texto em inglês. (Decisão do construtor do design.)

### Métricas

- Busca: context precision e context recall do Ragas.
- Resposta: faithfulness e correção face à referência (factual correctness ou equivalente na versão do Ragas em uso).
- Citação (métrica própria): fração das citações esperadas que aparecem na resposta, e fração das citações da resposta que existem nos trechos recebidos.
- Recusa: nas perguntas fora do assunto, a resposta recusou?
- Custo e tempo por pergunta: tokens e segundos da geração; custo do juiz à parte.

Dimensão da diferença: para cada métrica e configuração, o relatório mostra a média e um intervalo de confiança de 95% por bootstrap sobre as perguntas (sem custo de API), e a diferença entre configurações emparelhada por pergunta, também com intervalo. Diferenças cujo intervalo inclui zero são descritas como inconclusivas. Não há execuções repetidas por configuração; a variância entre execuções e o desenho estatístico de um teste A/B ficam para o terceiro projeto. (Decisão do construtor do design.)

O juiz do Ragas é o Claude Haiku 5.5. Gerar e julgar com a mesma família de modelos pode favorecer as respostas; o relatório diz isto como limitação.

Busca sem LLM: as métricas de busca usam as variantes do Ragas baseadas em ids de trechos de referência, ou métricas próprias (acerto entre os 5 primeiros e posição do primeiro acerto), a partir de `reference_citations`. A escolha fica ao construtor.

### Calibração do juiz

O autor quer saber se o juiz acerta. Depois da execução completa, o agente Claude Code que acompanha o projeto avalia às cegas uma amostra aleatória de cerca de 20 respostas (semente fixa, estratificada por configuração), com a mesma rubrica das métricas de resposta e sem ver as notas do Ragas. As notas ficam num ficheiro versionado. O relatório mostra a concordância entre as duas avaliações (percentagem de acordo e kappa de Cohen depois de binarizar as notas) e comenta os desacordos. Essa avaliação não é repetível por terceiros e o relatório diz isso.

### Execução e resultados

- `ingest`: descarrega, converte, divide e indexa, uma vez por estratégia de divisão; guarda em cache.
- `run --config X`: gera as respostas e guarda uma linha por pergunta (resposta, trechos, citações, tokens, tempo). Respostas já geradas para a mesma configuração e pergunta não são pedidas de novo.
- `evaluate --config X`: corre o Ragas e as métricas próprias sobre as respostas guardadas.
- `report`: junta as configurações num `REPORT.md` com tabela, gráfico (PNG), e uma secção de falhas que diz, para cada pergunta errada, se falhou a busca ou a resposta.
- Antes de qualquer comando pago, mostra a estimativa de custo e pede confirmação.

Os resultados da última execução completa ficam versionados, para o relatório ser legível no GitHub sem correr nada. Podem conter excertos literais dos documentos (trechos recuperados, contextos de referência), sempre com a fonte. São textos oficiais: atos da UE reutilizáveis com indicação da fonte (Decisão 2011/833/UE), textos legislativos portugueses fora da proteção do direito de autor, orientações da EBA reutilizáveis com atribuição. O construtor confirma a política de reutilização de cada fonte; se alguma não permitir, guarda só ids e hashes para esses trechos. Os documentos completos continuam fora do repositório, por tamanho e para cada um vir sempre da fonte oficial. (Decisão do construtor do design.)

### Testes

Unitários, sem rede nem API, no CI do GitHub Actions: divisão por artigo sobre excertos guardados como fixtures, leitura das configurações, métrica de citação, montagem do relatório a partir de resultados de exemplo, e a cache de respostas.

### Documentação

README em inglês e `README.pt.md`: o que o projeto mostra, como correr, o relatório mais recente, limitações (juiz da mesma família, corpus pequeno, referências revistas por uma pessoa) e próximos passos (painel de experiências, teste A/B). Nota de que não é aconselhamento jurídico.

## Deixado ao construtor

- Versões das bibliotecas e organização das pastas.
- BM25 ou vetores esparsos do BGE-M3 na busca híbrida.
- Qdrant local ou em Docker.
- Formato exato dos ficheiros de resultados.
- Parâmetros da divisão por tamanho fixo, se os de cima se mostrarem maus.
