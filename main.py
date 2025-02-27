from dotenv import load_dotenv
load_dotenv()
from graph.graph_flow import app

if __name__ == "__main__":
    print(app.invoke(input={"question":" give me an example for different prompting techniques"}))