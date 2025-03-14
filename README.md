# DEEP101 - Aprendizagem Profunda
Conhecer os conceitos fundamentais de aprendizado profundo, permitindo que os discentes possuam conhecimentos necessários para o aprofundamento em qualquer campo da área e que possam desenvolver métodos, ferramentas e aplicações inteligentes.

## Professores
Prof. Dr. Saulo Oliveira [Lattes](http://lattes.cnpq.br/9883694006602467) | [Google Scholar](https://scholar.google.com.br/citations?user=rRTkRcAAAAAJ)
e-mail: saulo[dot]oliveira[at]ifce.edu.br

## Informações
**Horário**: Às segundas, das 07:30 às 11:30.

**Local**: Sala 204, Bloco de Pós-graduação. Instituto Federal de Educação, Ciência e Tecnologia do Ceará | *campus* Fortaleza.

**Endereço**: Avenida Treze de Maio, nº 2081 - Benfica - CEP: 60040-215 - Fortaleza/CE.

## Pré-requisitos
Apesar das disciplinas do PPGCC não exigirem ```pré-requisitos```, o conhecimento das seguintes ferramentas será de grande valia para acelarar a curva de aprendizado durante a disciplina.

| 🐍  Python (Numpy & PyTorch)                                  | 🔢  Álgebra Linear                                            | 🧮 Cálculo (Derivadas)                                        |
| :----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Todas as atividades serão em Python e usaremos o formato de Notebooks do Jupyter para entrega. | Usaremos transposição de matriz, inversa e operações algébricas com expressões de matrizes. | Você precisará obter uma derivada e maximizar uma função descobrindo onde a derivada = 0. |

## Como vai ser a nota?
A avaliação da disciplina é qualitativa e visa o caminho da aprendizagem. Assim, ao invés de um valor número na escala 0-10, uma letra é atribuída. Esta letra indica um conjunto de fatores que são observados ao se realiazar as atividades avaialiativas durante a disciplina. A seguir, nos cards abaixo, são listados os fatores observados e a letra associada a cada um deles.

| A                                                            | B                                                            | C                                                            | D                                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| $\bullet$ Compreendeu por completo a proposta da atividade;<br/>$\bullet$ O trabalho está ótimo e completo;<br/>$\bullet$ O discente está pronto para o próximo assunto. | $\bullet$ Compreendeu a maior parte da proposta da atividade;<br/>$\bullet$ O trabalho está bom e completo;<br/>$\bullet$ O discente tem condições de progredir. | $\bullet$ Compreendeu parte da proposta da atividade;<br/>$\bullet$ O trabalho está incompleto;<br/>$\bullet$ O discente precisa de auxílio para compreender o assunto. | $\bullet$ Não compreendeu por completo a proposta da atividade;<br/>$\bullet$ O trabalho precisa ser refeito;<br/>$\bullet$ O discente não tem condições objetivas de progredir. |

Além do resultado da avaliação qualitativa, cada atividade avaliativa possui um peso associado. Em posse de cada avaliação e seu respectivo peso, a nota final que é a que vai para o histórico é calculada conforme a seguinte fórmula:

$$\text{Nota final} = \dfrac{\sum_{a_i ~\in ~\mathcal{A}} a_i * w_i }{\sum_{a_i ~\in ~\mathcal{A}} a_i} \text{ em que } a_i=$$ 

Observe um caso ilustrativo:

|          | Prova (2) | Simulação (1) | Relatório (1) | Seminário (1) | Nota final |
| -------- | :-------: | :-----------: | :-----------: | :-----------: | :--------: |
| Huguinho |     A     |       A       |       B       |       A       |    9,5     |
| Zezinho  |     B     |       B       |       B       |       C       |    7,0     |
| Luizinho |     C     |       A       |       B       |       B       |    8,0     |
| Donald   |     B     |       D       |       C       |       C       |    4,5     |

<small>Por questões éticas, com a finalidade de preservar a verdadeira identidade dos discentes, ocultaremos os nomes dos alunos. Assim, a identificação será por meio do número de matrícula, só que de forma truncada.</small>

## Cronograma e Conteúdo Programático

Este é o plano de estudos da iteração de outono de 2020 do curso.

| Tipo      | Data       | Descrição                                               | Material                                                     |
| --------- | ---------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| Aula      | 06/03/2023 | Introdução à aprendizagem de máquina                    | [Slides](slides/dl_00.pdf) • [Artigo](atividades/atividade-01.pdf) |
| Aula      | 13/03/2023 | Perceptron                                              | [Slides](slides/dl_01_ml.pdf) • [Atividade classificação](atividades/atividade-01.pdf) |
| Aula      | 20/03/2023 | Adaline                                                 | [Slides](slides/dl_03_adaline.pdf) • [Atividade regressão](atividades/atividade-01.pdf) |
| Aula      | 27/03/2023 | Perceptron multicamadas (MLP)                           | [Slides](slides/dl_03_adaline.pdf) • [Atividade regressão](atividades/atividade-01.pdf) |
| Aula      | 03/04/2023 | Otimizadores                                            | [Slides](slides/dl_03_adaline.pdf) • [Atividade regressão](atividades/atividade-01.pdf) |
| Prova     | 17/04/2023 | Perceptron, Adaline e MLPs                              | [Slides](slides/dl_03_adaline.pdf) • [Atividade regressão](atividades/atividade-01.pdf) |
| Seminário | 08/05/2023 | Desafio MLP                                             | [Slides](slides/dl_03_adaline.pdf) • [Atividade regressão](atividades/atividade-01.pdf) |
| Aula      | 05/05/2023 | Redes convolucionais                                    | [Slides](slides/dl_03_adaline.pdf) • [Atividade regressão](atividades/atividade-01.pdf) |
| Aula      | 08/05/2023 | Redes convolucionais famosinhas                         | [Slides](slides/dl_03_adaline.pdf) • [Atividade regressão](atividades/atividade-01.pdf) |
| Seminário | 08/05/2023 | Desafio CNN                                             | [Slides](slides/dl_03_adaline.pdf) • [Atividade regressão](atividades/atividade-01.pdf) |
| Aula      | 08/05/2023 | Redes autocodificadoras e Redes adversárias generativas | [Slides](slides/dl_03_adaline.pdf) • [Atividade regressão](atividades/atividade-01.pdf) |
| Aula      | 08/05/2023 | Redes recorrentes                                       | [Slides](slides/dl_03_adaline.pdf) • [Atividade regressão](atividades/atividade-01.pdf) |
| Prova     | 17/04/2023 | Solução de problemas da indústria                       | Nenhum.                                                      |
| Seminário | 08/05/2023 | Desafio RNN/GAN/VAE                                     | [Slides](slides/dl_03_adaline.pdf) • [Atividade regressão](atividades/atividade-01.pdf) |

<small>Cronograma básico. Ele pode ser alterado a qualquer momento por eventos diversos.</small>

## Diretrizes da Sessão de Pôsteres

O objetivo de um pôster é dar ao espectador um resumo rápido do trabalho, que não apenas atrai seu interesse, mas também o motiva a aprender mais sobre o projeto. Para uma introdução à criação de pôsteres acadêmicos, consulte [Guia de postêres](https://guides.nyu.edu/posters) da Universidade de Nova Iorque. Aqui está um exemplo de estrutura que pode ser útil:

| Problema                                                     | Antecedentes                                                 | Métodos                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| O que você está tentando resolver? Consulte os desafios atuais e por que seu trabalho é significativo. | Configuração do problema + notação; talvez trabalhos anteriores. | Explique suas contribuições técnicas; números podem realmente ajudar a entender especialmente para um modelo neural! |

| Experimentos                                                 | Análise                                                      | Conclusão e referências                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Qual é a tarefa? Quais são as entradas e saídas do modelo? O que são as resultados? Compare com as linhas de base; para resultados, um gráfico pode dizer muito mais do que uma tabela do mesmo tamanho! | Forneça uma análise aprofundada de certos casos de interesse ou contextualize seus resultados na literatura existente! Adicione alguns gráficos/exemplos/visualizações. | Tire brevemente algumas conclusões do trabalho.<br/>Se o espaço permitir, considere adicionar 1-3 referências muito importantes (não faça mais do que isso!) |

