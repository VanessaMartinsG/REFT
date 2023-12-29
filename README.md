# REFT - Reconhecimento de Emoções através de Fala e Texto

O REFT é um sistema integrado que combina o Reconhecimento de Emoções na Fala (REF) e o Reconhecimento de Emoções no Texto (RET) para fornecer uma análise abrangente das emoções expressas em áudio e texto.

## Descrição

Este projeto integra duas partes principais: o REF, que analisa as emoções em fala, e o RET, que analisa as emoções em texto. O REFT utiliza o Google Speech Recognition para transcrever a fala para texto e, em seguida, aplica modelos de aprendizado de máquina treinados para prever as emoções no texto e na fala. Os resultados são combinados para fornecer uma avaliação final ponderada das emoções.

## Requisitos

- Python 3
- Bibliotecas Python: speech_recognition e as bibliotecas necessárias para REF e RET.

## Configuração

```python

	# 1. Clone este repositório:
	git clone git@bitbucket.org:bahiartathome/reft.git
	cd REFT


	# 2. Instale as dependências:
	pip install -r requirements.txt
	
	
	# 3. Execute o script principal:
	python main.py
		
```
	
## Utilização

- Inicie o script e fale para o microfone. O sistema analisará as emoções expressas na fala e no texto.

## Resultados

Os resultados da análise, incluindo o texto reconhecido, as emoções detectadas na fala e no texto, e a emoção final ponderada, serão exibidos no console e armazenados em um arquivo CSV para referência futura.

### Atenção:

Para ter acesso aos modelos treinados utilizados nesta pesquisa (REF e RET), consulte o perfil da pesquisadora no [Hugging Face REFT](https://huggingface.co/VanessaMartinsG).

## Contribuição

Sinta-se à vontade para contribuir, relatar problemas ou fazer sugestões para melhorar este projeto. Aceitamos contribuições por meio de pull requests.

## Informação Adicional

Este trabalho foi desenvolvido como resultado da tese de graduação da aluna Vanessa Martins da Universidade do Estado da Bahia (UNEB). Para mais detalhes, a monografia completa pode ser consultada no repositório da universidade.

### Contato:

E-mail: nessa.martinsgomes@gmail.com
