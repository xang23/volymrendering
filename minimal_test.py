import vtk
from PyQt5 import QtWidgets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import sys

class MinimalTest(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Minimal VTK Test")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        
        # Create VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor()
        layout.addWidget(self.vtk_widget)
        
        # Create a simple renderer with a cube
        renderer = vtk.vtkRenderer()
        
        # Create a cube
        cube = vtk.vtkCubeSource()
        cube.SetXLength(1.0)
        cube.SetYLength(1.0)
        cube.SetZLength(1.0)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 0)  # Red
        
        renderer.AddActor(actor)
        renderer.SetBackground(0.1, 0.1, 0.1)
        renderer.ResetCamera()
        
        # Add renderer to VTK widget
        self.vtk_widget.GetRenderWindow().AddRenderer(renderer)
        
        # Initialize and start
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
        
        print("Minimal test should show a red cube")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MinimalTest()
    window.show()
    sys.exit(app.exec_())