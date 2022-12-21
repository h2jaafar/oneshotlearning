import dearpygui.dearpygui as dpg
import dearpygui.demo as demo
from main import MainRunner



class MainWindow:
    def __init__(self):
        self.configurationParameters = {
        "Layers_A": 0,
        "Kernel_A": 0,
        "Layers_B": 0,
        "Kernel_B": 0,
        "Layers_C": 0,
        "Kernel_C": 0,
        "Layers_D": 0,
        "Kernel_D": 0,
        "Layers_E": 0,
        "epochs": 10,
        "batch_size": 32,
        "lr": 0.00005,
        "conv activation": "relu",
        "final activation": "sigmoid",
        "savePath": "./dataset/",
        "load_data": False,
        "analyze": False,
        "ytf": True,

    }

    def _gui_open_image(self, sender, app_data):
        print('OK was clicked.')
        print("Sender: ", sender)
        print("App Data: ", app_data)

    def cancel_callback(sender, app_data):
        print('Cancel was clicked.')
        print("Sender: ", sender)
        print("App Data: ", app_data)

    ## global items 


    def init_network_callback(self):
        self.configurationParameters["Layers_A"] = dpg.get_value("layers_A")
        self.configurationParameters["Kernel_A"] = dpg.get_value("kernel_A")
        self.configurationParameters["Layers_B"] = dpg.get_value("layers_B")
        self.configurationParameters["Kernel_B"] = dpg.get_value("kernel_B")
        self.configurationParameters["Layers_C"] = dpg.get_value("layers_C")
        self.configurationParameters["Kernel_C"] = dpg.get_value("kernel_C")
        self.configurationParameters["Layers_D"] = dpg.get_value("layers_D")
        self.configurationParameters["Kernel_D"] = dpg.get_value("kernel_D")
        self.configurationParameters["Layers_E"] = dpg.get_value("layers_E")
        self.configurationParameters["epochs"] = dpg.get_value("epochs")
        self.configurationParameters["batch_size"] = dpg.get_value("batch_size")
        self.configurationParameters["lr"] = dpg.get_value("lr")
        self.configurationParameters["conv activation"] = dpg.get_value("conv activation")
        self.configurationParameters["final activation"] = dpg.get_value("final activation") 
        self.configurationParameters["savePath"] = dpg.get_value("savePath") 
        self.configurationParameters["load_data"] = dpg.get_value("load_data") 
        self.configurationParameters["analyze"] = dpg.get_value("anaylze") 
        self.configurationParameters["ytf"] = dpg.get_value("ytf") 

        mr = MainRunner(configs=self.configurationParameters)
        print("Initialized Network")

    def runWindow(self):

        dpg.create_context()
        dpg.create_viewport(title="One Shot Learning", resizable=True, clear_color=[70, 80, 90, 255])
        dpg.setup_dearpygui()


        width, height, channels, data = dpg.load_image("./docs/architecture_small.png")

        with dpg.texture_registry(show=False):
            dpg.add_static_texture(width=width, height=height, default_value=data, tag="texture_tag")

        dpg.add_file_dialog(
        directory_selector=True, show=False, callback=self._gui_open_image, tag="file_dialog_id",
        cancel_callback=self.cancel_callback)
        # dpg.show_metrics()
        # dpg.show_style_editor()
        # demo.show_demo()


        with dpg.window(label="Main Window", autosize=True):
            dpg.add_text("Configuration")
            dpg.add_input_int(label="Layers in A", default_value=64, width = 128, tag="layers_A")
            dpg.add_input_int(label="Kernel Size in A", default_value=10, width =128, tag="kernel_A")
            dpg.add_input_int(label="Layers in B", default_value=128, width = 128, tag="layers_B")
            dpg.add_input_int(label="Kernel Size in B", default_value=7, width =128, tag="kernel_B")
            dpg.add_input_int(label="Layers in C", default_value=128, width = 128, tag="layers_C")
            dpg.add_input_int(label="Kernel Size in C", default_value=4, width =128, tag="kernel_C")
            dpg.add_input_int(label="Layers in D", default_value=256, width = 128, tag="layers_D")
            dpg.add_input_int(label="Kernel Size in D", default_value=4, width =128, tag="kernel_D")
            dpg.add_input_int(label="Layers in E", default_value=4096, width = 128, tag="layers_E")
            dpg.add_input_int(label="epochs", default_value=10, width = 128, tag="epochs")
            dpg.add_input_int(label="batch size", default_value=32, width = 128, tag="batch_size")
            dpg.add_input_float(label="learn rate", default_value=0.00005, width = 128, step=0.00001, format="%.5f", tag="lr")
            dpg.add_combo(items=(["relu","leaky-relu", "sigmoid"]), label="conv activation", default_value="leaky-relu", width = 128, tag="conv activation")
            dpg.add_combo(items=(["sigmoid"]), label="final layer activation", default_value="sigmoid", width = 128, tag="final activation")
            dpg.add_checkbox(label="Include YoutubeFaces dataset", default_value=False, tag="ytf")
            dpg.add_checkbox(label="reload data", default_value=False, tag="load_data")
            dpg.add_input_text(label="training file name", default_value="train_10", tag="savePath")
            dpg.add_checkbox(label="analyze after", default_value=False, tag="analyze")


            # assign values

            
            dpg.add_image("texture_tag")
            # dpg.add_button(label="Directory Selector", callback=lambda: dpg.show_item("file_dialog_id"))

            # Init Network
            dpg.add_button(label="Initialize Network", callback=self.init_network_callback)
            dpg.add_spacer()
            dpg.add_text("Train Network")

        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

if __name__ == "__main__":
   mw = MainWindow()
   mw.runWindow()
