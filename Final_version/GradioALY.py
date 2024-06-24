import gradio as gr

with gr.Blocks( title="AVATARES") as demo:
    gr.Markdown("## ""![](file/Gradio/logo_lis.svg) AVATARES""", elem_classes="cabecero")
    gr.Markdown("""ALi guapisima""", elem_classes="cabecero")
    with gr.Tabs() as tabs:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Textbox(label="Hola")
                gr.Textbox()
                gr.Markdown("""ALi guapisima""", elem_classes="cabecero")
                gr.Textbox()
                gr.Textbox()
                gr.Textbox()
                gr.Textbox()
        with gr.Row():
            with gr.Column(scale=1):
                gr.Textbox()
            with gr.Column(scale=1):
                gr.Textbox()
            with gr.Column(scale=1):
                gr.Textbox()
        with gr.Row():
            gr.Button("Botoón!!!", elem_classes="")

demo.launch(favicon_path = "Gradio/favicon.png")