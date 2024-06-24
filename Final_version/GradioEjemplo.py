import gradio as gr
import os
import altair as alt
import numpy as np
import pandas as pd
from vega_datasets import data

def func(slider_1, slider_2):
    return slider_1 * 5 + slider_2

def combine(a, b):
    return a + " " + b


def mirror(x):
    return x

def make_plot(plot_type):
    if plot_type == "scatter_plot":
        cars = data.cars()
        return alt.Chart(cars).mark_point().encode(
            x='Horsepower',
            y='Miles_per_Gallon',
            color='Origin',
        )
    elif plot_type == "heatmap":
        # Compute x^2 + y^2 across a 2D grid
        x, y = np.meshgrid(range(-5, 5), range(-5, 5))
        z = x ** 2 + y ** 2

        # Convert this grid to columnar data expected by Altair
        source = pd.DataFrame({'x': x.ravel(),
                            'y': y.ravel(),
                            'z': z.ravel()})
        return alt.Chart(source).mark_rect().encode(
            x='x:O',
            y='y:O',
            color='z:Q'
        )
    elif plot_type == "us_map":
        states = alt.topo_feature(data.us_10m.url, 'states')
        source = data.income.url

        return alt.Chart(source).mark_geoshape().encode(
            shape='geo:G',
            color='pct:Q',
            tooltip=['name:N', 'pct:Q'],
            facet=alt.Facet('group:N', columns=2),
        ).transform_lookup(
            lookup='id',
            from_=alt.LookupData(data=states, key='id'),
            as_='geo'
        ).properties(
            width=300,
            height=175,
        ).project(
            type='albersUsa'
        )
    elif plot_type == "interactive_barplot":
        source = data.movies.url

        pts = alt.selection(type="single", encodings=['x'])

        rect = alt.Chart(data.movies.url).mark_rect().encode(
            alt.X('IMDB_Rating:Q', bin=True),
            alt.Y('Rotten_Tomatoes_Rating:Q', bin=True),
            alt.Color('count()',
                scale=alt.Scale(scheme='greenblue'),
                legend=alt.Legend(title='Total Records')
            )
        )

        circ = rect.mark_point().encode(
            alt.ColorValue('grey'),
            alt.Size('count()',
                legend=alt.Legend(title='Records in Selection')
            )
        ).transform_filter(
            pts
        )

        bar = alt.Chart(source).mark_bar().encode(
            x='Major_Genre:N',
            y='count()',
            color=alt.condition(pts, alt.ColorValue("steelblue"), alt.ColorValue("grey"))
        ).properties(
            width=550,
            height=200
        ).add_selection(pts)

        plot = alt.vconcat(
            rect + circ,
            bar
        ).resolve_legend(
            color="independent",
            size="independent"
        )
        return plot
    elif plot_type == "radial":
        source = pd.DataFrame({"values": [12, 23, 47, 6, 52, 19]})

        base = alt.Chart(source).encode(
            theta=alt.Theta("values:Q", stack=True),
            radius=alt.Radius("values", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
            color="values:N",
        )

        c1 = base.mark_arc(innerRadius=20, stroke="#fff")

        c2 = base.mark_text(radiusOffset=10).encode(text="values:Q")

        return c1 + c2
    elif plot_type == "multiline":
        source = data.stocks()

        highlight = alt.selection(type='single', on='mouseover',
                                fields=['symbol'], nearest=True)

        base = alt.Chart(source).encode(
            x='date:T',
            y='price:Q',
            color='symbol:N'
        )

        points = base.mark_circle().encode(
            opacity=alt.value(0)
        ).add_selection(
            highlight
        ).properties(
            width=600
        )

        lines = base.mark_line().encode(
            size=alt.condition(~highlight, alt.value(1), alt.value(3))
        )

        return points + lines
    
with gr.Blocks(css="styles.css", title="0Accidentes") as demo:
    gr.Markdown("""![](file/logo_lis.svg)# 0Accidentes""", elem_classes="cabecero")

    with gr.Tab("Pestaña 1"):
        txt = gr.Textbox(label="Input", lines=2)
        txt_2 = gr.Textbox(label="Input 2")
        txt_3 = gr.Textbox(value="", label="Output")
        btn = gr.Button(value="Submit", elem_classes="button")
        btn.click(combine, inputs=[txt, txt_2], outputs=[txt_3])

        with gr.Row():
            im = gr.Image()
            im_2 = gr.Image()

        btn = gr.Button(value="Mirror Image", elem_classes="button")
        btn.click(mirror, inputs=[im], outputs=[im_2])

        gr.Markdown("## Text Examples")
        gr.Examples(
            [["hi", "Adam"], ["hello", "Eve"]],
            [txt, txt_2],
            txt_3,
            combine,
            cache_examples=True,
        )
        gr.Markdown("## Image Examples")
        gr.Examples(
            examples=[os.path.join(os.path.dirname(__file__), "lion.jpg")],
            inputs=im,
            outputs=im_2,
            fn=mirror,
            cache_examples=True,
        )
    with gr.Tab("Pestaña 2"):
            button = gr.Radio(label="Plot type",
                      choices=['scatter_plot', 'heatmap', 'us_map',
                               'interactive_barplot', "radial", "multiline"], value='scatter_plot')
            plot = gr.Plot(label="Plot")
            button.change(make_plot, inputs=button, outputs=[plot])
            demo.load(make_plot, inputs=[button], outputs=[plot])

    with gr.Tab("Pestaña 3"):
                input_files = gr.inputs.File(file_count="multiple", label="")
                text_button = gr.Button("Construir Asistente", elem_classes="button") #elem_classes te permite vincular el estilo del botón o campo al valor equivalente de la hoja de estilos
                text_output = gr.Textbox(show_label=False)
                #text_button.click(self.gui_trigger_build, [input_files], text_output)
    
    with gr.Tab("Pestaña 4"):
                embedding_model_options =["text-embedding-ada-002", 'Model 2', 'Model 3'] # embeddings in lllm_tools_lis....a bit thinking required since when loading from db works different... check examples.py
                embedding_model = gr.Dropdown(embedding_model_options, label="Embedding model choices")
                b1 = gr.Button("save", elem_classes="button")
                #b1.click(self.grab_b1,inputs=embedding_model, outputs=embedding_model)
                #https://discuss.huggingface.co/t/how-to-update-the-gr-dropdown-block-in-gradio-blocks/19231

                generative_model_options = ["gpt-4","gpt-35-turbo","text-davinci-003"] #test_model_name in llm_tools_lis
                generative_model = gr.Dropdown(generative_model_options, label="Generative model choices")
                b2 = gr.Button("save", elem_classes="button")
                #b2.click(self.grab_b2,inputs=generative_model, outputs=generative_model)

                generative_temperature_options = ['0', '0.01', '0.1'] #temperature in load_qa_chain
                generative_temperature = gr.Dropdown(generative_temperature_options, label="Generative temperature")
                b3 = gr.Button("save", elem_classes="button")
                #b3.click(self.grab_b3,inputs=generative_temperature, outputs=generative_temperature)

                context_restrictions_options = [True, False] #temperature in load_qa_chain
                context_restrictions = gr.Dropdown(context_restrictions_options, label="Context restrictions?")
                b4 = gr.Button("save", elem_classes="button")
                #b4.click(self.grab_b4,inputs=context_restrictions, outputs=context_restrictions)

    with gr.Tab("Pestaña 5"):  
        slider = gr.Slider(minimum=-10.2, maximum=15, label="Random Slider (Static)", randomize=True)
        slider_1 = gr.Slider(minimum=100, maximum=200, label="Random Slider (Input 1)", randomize=True, elem_classes="slider")
        slider_2 = gr.Slider(minimum=10, maximum=23.2, label="Random Slider (Input 2)", randomize=True)
        slider_3 = gr.Slider(value=3, label="Non random slider")
        btn = gr.Button("Run", elem_classes="button")
        btn.click(func, inputs=[slider_1, slider_2], outputs=gr.Number())


        

if __name__ == "__main__":
    demo.launch()

