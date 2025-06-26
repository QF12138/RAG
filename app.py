from flask import Flask, request, jsonify, render_template
from LLM分流加分句子输出_api import EnhancedRAG

app = Flask(__name__)

# 初始化 EnhancedRAG 模型（假设无需额外参数，如有需要请根据 local.py 调整）
rag = EnhancedRAG()


@app.route('/')
def index():
    # 渲染前端页面模板
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    # 获取 JSON 请求数据
    data = request.get_json()
    question = data.get('question', '')
    # 调用 EnhancedRAG 获取回答
    answer = rag.ask(question)
    # 返回 JSON 应答
    return jsonify({'answer': answer})


if __name__ == '__main__':
    # 运行 Flask 开发服务器
    app.run(debug=False)
