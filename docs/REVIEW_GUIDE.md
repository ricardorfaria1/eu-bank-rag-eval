# Guia de revisão do conjunto de teste

O comando `uv run eurag testset` gera `data/testset/candidates.csv`, com cerca de 80 perguntas criadas pelo Ragas. Só as linhas que uma pessoa revê entram na avaliação.

## Como rever

1. Abra `data/testset/candidates.csv` numa folha de cálculo (Excel, Numbers ou Google Sheets) como CSV UTF-8.
2. Para cada linha, leia `user_input` (a pergunta), `reference` (a resposta certa) e `reference_contexts` (os trechos de onde a resposta veio).
3. Confirme a referência no texto oficial, sobretudo números, prazos e artigos.
4. Mude a coluna `origin`:
   - `reviewed` se a pergunta e a referência estão certas;
   - `reviewed`, depois de corrigir `user_input` ou `reference`, e escreva em `review_note` o que mudou;
   - `rejected` se a pergunta é confusa, trivial, ou a referência não se aproveita.
5. Mantenha a `category` certa: `single_doc`, `multi_doc`, `cross_language`, `revoked`, `cross_supervisor` ou `out_of_scope`.
6. Guarde como `data/testset/reviewed.csv`, em CSV UTF-8, sem mudar a ordem das colunas.

O objetivo é ficar com cerca de 40 perguntas revistas. Junte-as aos casos escritos à mão com:

```bash
uv run eurag testset --merge
```

## O que procurar

- Prazos e números trocados (28 de fevereiro contra 31 de março, por exemplo).
- Referências que citam uma norma revogada como se estivesse em vigor.
- Perguntas que só fazem sentido para quem viu o trecho ("o que diz este artigo?").
- Traduções para português do Brasil em vez de português europeu.
