import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from streamlit.runtime.caching import cache_resource
import pandas as pd
import torch
import matplotlib.pyplot as plt

@cache_resource
def load_mapping():
    df=pd.read_excel('./label_mapping.xlsx')
    dict_data = df.set_index('label')['处理部门'].to_dict()
    return dict_data

@cache_resource(show_spinner=False)
# 使用st.cache_resource装饰器缓存模型和tokenizer的加载过程
def load_model_and_tokenizer():
    #with st.spinner('模型加载中...'):
    model_path = './model'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('./model')
    return model, tokenizer



def process_text(input_text, model, tokenizer):
    # 实现处理文本的逻辑，包括使用tokenizer进行编码等
    encoded_inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")
    # encoded_inputs={k: v.cuda() for k, v in encoded_inputs.items()}
    # 使用模型进行推理
    with torch.no_grad():
        outputs = model(**encoded_inputs)
    
    # 应用softmax来转换为概率
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # 获取最大概率的索引作为预测结果
    predicted_label = torch.argmax(probabilities, dim=-1).item()  # 转换为Python整数
    # return f"模型处理后的文本结果: {encoded_inputs['input_ids']}"  # 示例逻辑
    return encoded_inputs,predicted_label,probabilities

def process_csv(uploaded_file, model, tokenizer):
    # 处理CSV文件的逻辑
    df = pd.DataFrame(pd.read_excel(uploaded_file,index_col=False))
    df['preLabel']=0
    df['result']=''
    for i in range(10):
        input_text=df.loc[i, 'new_content']
        with torch.no_grad():
            inputs = tokenizer(input_text, truncation=True, padding='max_length', max_length=512, return_tensors="pt")
            #inputs = {k: v.cuda() for k, v in inputs.items()}
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=-1)
            df.loc[i, 'preLabel']=pred.item()
            dict_data=load_mapping()
            df.loc[i, 'result']=dict_data.get(pred.item())
    return df  # 示例返回DataFrame头部
def create_pie_chart(data, labels):
    """根据给定的数据和标签创建饼图"""
    fig, ax = plt.subplots()
    ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    return fig

 # 初始化session_state
if 'predicted_label' not in st.session_state:
    st.session_state['predicted_label']=""
if 'input_ids' not in st.session_state:
    st.session_state['input_ids']={}
if 'processed_text' not in st.session_state:
    st.session_state['processed_text'] = "" 
if 'probabilities' not in st.session_state:
    st.session_state['probabilities']={}

def main():    
    st.markdown(
        "<h1 style='text-align: center;'>🤗 模型指派过程演示</h1>", 
        unsafe_allow_html=True
    )
    loading_palceholder=st.empty()
    with loading_palceholder.container():
        with st.spinner('模型加载中....'):
            model, tokenizer = load_model_and_tokenizer()
            dict_data=load_mapping()
    loading_palceholder.empty()
    st.success("模型及Tokenizer已成功加载！！！！！")
    
    input_type = st.radio("选择输入类型", ("工单内容", "工单内容文件(.xlsx)"))
    if input_type == "工单内容":
        input_text = st.text_area("请输入文本")
        if st.button("处理工单内容"):
            encoded_inputs,predicted_label,_= process_text(input_text, model, tokenizer)
            st.session_state.processed_text = encoded_inputs
            st.session_state['input_ids']=encoded_inputs['input_ids'].tolist()
        st.code(f'编码后的工单内容：{st.session_state.input_ids}')
        if st.button("输入模型"):
            _,predicted_label,probabilities= process_text(input_text, model, tokenizer)
            st.session_state.predicted_label=predicted_label
            st.session_state.probabilities=probabilities.tolist()
        st.code(f'各个指派部门概率：{st.session_state.probabilities}')
        st.code(f'指派部门代码：{st.session_state.predicted_label}')
        if st.button("指派部门映射"):
            st.code(f'指派部门：{dict_data[st.session_state.predicted_label]}')
    else:
        uploaded_file = st.file_uploader("上传xlsx文件", type=["xlsx"])
        if uploaded_file is not None:
            df = pd.DataFrame(pd.read_excel(uploaded_file,index_col=False))
            st.dataframe(df)
            if st.button("输入模型"):
                result = process_csv(uploaded_file, model, tokenizer)
                with st.expander('指派结果展示：', expanded=True):
                    st.dataframe(result)
                with st.expander('指派结果饼图：', expanded=True):
                    correct = result['label'] == result['preLabel']
                    correct_count = correct.sum()
                    total_count = len(result)
                    accuracy = correct_count / total_count
                    error_rate = (total_count - correct_count)/ total_count
                    data=[accuracy,error_rate]
                    labels = ['true', 'false']
                    # 创建饼图
                    fig = create_pie_chart(data, labels)
                    # 展示饼图
                    st.pyplot(fig)
        
if __name__ == "__main__":
    main()
