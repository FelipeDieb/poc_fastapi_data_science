# poc_fastapi_data_science

Instruções para iniciar o servidor
1) Abrir o cmd e ir até o caminho da aplicacao
2) Criar um ambiente env com o comando abaixo(Este comando so precisa executar a primeira vez, nas proximas pular para a etapa 3):
	- python -m venv ./venv
3) Ativar o ambiente env com o comando abaixo:
	- venv\Scripts\activate.bat ou c:/Users/seu_user/path_ate_o_projeto/XAI/API/venv/Scripts/Activate.ps1
4) Instalar pacotes necessários:
	- pip install -U -r requirements.txt (se sugir erros add : --force-reinstall)
5) Subir o servidor com o comando :
	- uvicorn main:app --reload
