<div style="text-align: center;">
<img src="https://github.com/sauloafoliveira/ppgcc/blob/main/classes/deep_learning/2023_1/ppgc_logo.png" alt="PPGCC logo" style="width: 60%" />
</div>

<h1 class="t">Aprendizagem Profunda – 2023.1</h1>

**Prof. Dr. Saulo Oliveira**

**Data de Entrega:** 22 de maio de 2023.

**Meio de Entrega:** Seminário.


## Dataset: LIVE Image Quality Assessment Database (release 1)

O dataset Live da Universidade do Texas (UTexas) [link aqui](https://live.ece.utexas.edu/research/Quality/subjective.htm) possui $140$ MB de tamanho. Nele, $29$ imagens coloridas RGB de 24 bits/pixel de alta resolução (normalmente $768 \times 512$) foram comprimidas usando JPEG com diferentes taxas de compressão para produzir um banco de dados de $204$ imagens, $29$ das quais foram as imagens originais (descompactadas). 

> **A senha do zip é <mark>livequality2002</mark>.**


Um dos estudos realizado neste conjunto de dados (release 01) aconteceu em duas sessões. A <kbd>sessão 01</kbd> contém imagens ```img1.bmp``` a ```img116.bmp``` e a <kbd>sessão 02</kbd> contém imagens ```img117.bmp``` a ```img233.bmp```. As taxas de bits escolhidas de forma que a distribuição resultante da qualidade as pontuações para as imagens comprimidas foram aproximadamente uniformes. 

A cada pessoa foi mostrada as imagens aleatoriamente. Os observadores foram solicitados a fornecer sua percepção de qualidade em um escala categórica ($1$ a $5$), como segue:
- ```Bad```;
- ```Poor```;
- ```Fair```;
- ```Good```, e
- ```Excellent```. 
 
A escala foi então convertida em $1-100$ linearmente. O teste foi feito em duas sessões com cerca de metade das imagens em cada sessão.


O arquivo de informações [jpeginfo.txt](#) contém uma lista que descreve como a base de dados foi criada.
Cada linha é uma entrada separada no banco de dados de imagens:

> ```<Imagem de origem> <Imagem de destino> <Taxa de bits atingida>```

Uma taxa de bits com valor $0$ significa uma cópia sem perdas do arquivo de origem!


Cada imagem no banco de dados é seguida pelas pontuações atribuídas a ela pelos diferentes observadores. Uma pontuação de $0$ significa que o sujeito pulou a imagem.

Os arquivos ```.mat``` (um para cada sessão) possuem as seguintes variáveis:
- ```mmt``` é a pontuação processada média para a imagem;
- ```mst``` é o desvio padrão das pontuações processadas para a imagem;
- ```br``` é a taxa de bits usada para essa imagem. Possuir ```br == 0``` significa SEM PERDA!
- ```scores(i,:)``` é a matriz de pontuação processada para a imagem i. Uma pontuação de zero implica que foi ignorado ou removido na etapa de remoção de *outlier*;
- ```scores``` contém apenas pontuações processadas, o que significa que alguns assuntos podem ter sido removidos.

Para ler os arquivos no formato do ```Matlab``` em ```Python``` (os arquivos são lidos como ```dict```s), basta importar o pacote ```scipy.io```:

```python
import scipy.io
mat = scipy.io.loadmat('file.mat')
```

<small>
Fonte: <a href="https://stackoverflow.com/questions/874461/read-mat-files-in-python">https://stackoverflow.com/questions/874461/read-mat-files-in-python</a>.
</small>

## O que é o trabalho?

A equipe precisará propor uma rede convolucional com uma das arquiteturas anteriormente descritas e estimar a qualidade tal qual como pessoas, isto é, a rede deverá se comportar como uma pessoa e gerar um conjunto de escores compatíveis.

A avaliação do resultado deverá contar com quatro métricas de desempenho que correlacionam as saída da rede e os valores subjetivos (de pessoas), a saber, 

- O Erro quadrático médio (RMSE);
- O coeficiente de correlação linear de Pearson (LLC);
- O coeficiente de correlação de ordem de classificação de Spearman (SRCC); e por fim,
- A razão de outlier (OR).

```python
import torch
from torchmetrics import SpearmanCorrCoef, PearsonCorrCoef, MeanSquaredError

def outlier_rate(scores):
    std, mean = torch.std_mean(scores)

    outilers = torch.logical_or(
            scores < (mean - 2 * std),
            scores > (mean + 2 * std)
    )    
    
    return torch.mean(outilers.double())
```
 
 De modo que o desempenho deve ser avaliado conforme as seguinte métricas:
 ```python
 import torch
 from torchmetrics import SpearmanCorrCoef, PearsonCorrCoef, MeanSquaredError
 
 y_pred = torch.tensor([3, -0.5, 2, 7]) # suponha que veio do modelo
 y = torch.tensor([2.5, 0.0, 2, 8]) # suponha que veio dos arquivos do dataset

 pearson = PearsonCorrCoef()
 spearman = SpearmanCorrCoef()
 mse = MeanSquaredError()
 
 print("Root Mean Square Error (RMSE):", mse(y_pred, y) ** 0.5)
 print("Pearson (LLC):", pearson(y_pred, y))
 print("Spearman (SRCC):", spearman(y_pred, y))
 print("Outlier Rate (OR):", outlier_rate(y_pred))

 ```
 
## Regras gerais

A seguir, as regras que delimitam os aspectos desse projeto:

- Utilizem os dados da <kbd>sessão 01</kbd> para treino e os da <kbd>sessão 02</kbd> para teste;
- Não pode usar rede treinada, ou seja, o treinamento tem que partir de vocês;
- Não pode fazer aprendizagem por transferência;
- Não pode pré-processar a imagem, com exceção de transformações de escala dos pixels ou redimensionamento da entrada;
- O treinamento tem de ser via Google Colab.
 
## Comparação

 Abaixo, condenso em uma tabela a avaliação de alguns algoritmos do tipo *referência completa* (FR):
 

 
| Tipo   | Algoritmo    | RMSE   | LCC    | SRCC   | OR     |
| ------ | ------------ | ------ | ------ | ------ | ------ |
| FR     | [FSIM](http://doi.org/10.1109/TIP.2011.2109730) (2011)         | 7.6780 | 0.9597 | 0.9634 | -      |
| FR     | VSNR         | 7.390  | 0.889  | 0.889  | -      |
| FR     | [MAD](https://doi.org/10.1117/1.3267105) (2010)         | -      | 0.9683 | 0.9675 | 41.46% |
| FR     | [IW-SSIM](https://doi.org/10.1109/TIP.2010.2092435) (2011)     | 8.347  | 0.9522 | 0.9567 | -      |
| FR     | [GS](https://doi.org/10.1109/TIP.2011.2175935) (2012)          | 9.0376 | 0.9437 | 0.9554 | -      |
| FR     | HDR-VDR-2    | 8.5    | -      | 0.9544 | -      |
| FR     | [VSI](10.1109/TIP.2014.2346028) 2014          | 8.6816 | 0.9482 | 0.9524 | -      |
| FR     | [Fast-MS-SSIM](https://doi.org/10.1007/s11554-010-0170-9) (2010) | 8.0787 | 0.9369 | 0.9409 | -      |
| FR     | [SFF](https://doi.org/10.1109/TIP.2013.2266579) (2013)          | 7.3460 | 0.9632 | 0.9649 | 0.0655 |
| FR     | SR-SIM       | 8.0811 | 0.9553 | 0.9618 | -      |

 <small>
  <strong>Fontes</strong>: <a href="https://live.ece.utexas.edu/research/Quality/evaluation.htm">LIVE Image Quality Assessment Database Algorithm Comparison Results</a>. Acessado em 12 de maio de 2023.
 </small>
 
## Prêmio

Os membros da equipe vencedora (a arquitetura com melhor desempenho nas métricas acima) terão a atividade da disciplina que possuir o menor desempenho  **eliminada** da **nota final** da disciplina. Ao passo que ao não entregar um trabalho *consistente*, a nota desta atividade refletirá a qualidade da entrega.

# Referências

- Aditya Chatterjee. **Evolution of CNN Architectures**: LeNet, AlexNet, ZFNet, GoogleNet, VGG and ResNet. https://iq.opengenus.org/evolution-of-cnn-architectures/. 2023, Acessado em mai, 2023.
- Bharath Raj. **A Simple Guide to the Versions of the Inception Network**. https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202, 2018. Acessado em 06 de maio de 2023.

- H. R. Sheikh, M. F. Sabir, A. C. Bovik. **A Statistical Evaluation of Recent Full Reference Quality Assessment Algorithms**, IEEE Transactions on Image Processing, vol. 15, no. 11, pp. 3440-3451, Nov. 2006.
- HE, Kaiming et al. **Deep residual learning for image recognition**. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
- KRIZHEVSKY, Alex; SUTSKEVER, Ilya; HINTON, Geoffrey E. **ImageNet classification with deep convolutional neural networks**. Communications of the ACM, v. 60, n. 6, p. 84-90, 2017.
- RONNEBERGER, Olaf; FISCHER, Philipp; BROX, Thomas. **U-net: Convolutional networks for biomedical image segmentation**. In: Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer International Publishing, 2015. p. 234-241.

- SIMONYAN, Karen; ZISSERMAN, Andrew. **Very deep convolutional networks for large-scale image recognition**. arXiv preprint arXiv:1409.1556, 2014.
- SZEGEDY, Christian et al. Going deeper with convolutions. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
