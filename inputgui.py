import sys
import pandas as pd

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QLabel,
    QFileDialog
)

CSV_FILE = "inputs_vertical_sample.csv"


class InputEditor(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Rocket Simulation Inputs")
        self.resize(1000, 800)

        self.layout = QVBoxLayout()

        self.title_label = QLabel("Rocket Simulation Parameters")
        self.layout.addWidget(self.title_label)

        self.table = QTableWidget()
        self.layout.addWidget(self.table)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_csv)
        self.layout.addWidget(self.save_button)

        self.setLayout(self.layout)

        self.load_csv()

    def load_csv(self):
        try:
            self.df = pd.read_csv(CSV_FILE)

            self.table.setRowCount(len(self.df))
            self.table.setColumnCount(3)
            self.table.setHorizontalHeaderLabels(
                ["Name", "Value", "Unit"]
            )

            for row in range(len(self.df)):
                name_item = QTableWidgetItem(str(self.df.iloc[row]["name"]))
                value_item = QTableWidgetItem(str(self.df.iloc[row]["value"]))
                unit_item = QTableWidgetItem(str(self.df.iloc[row]["unit"]))

                self.table.setItem(row, 0, name_item)
                self.table.setItem(row, 1, value_item)
                self.table.setItem(row, 2, unit_item)

            self.table.resizeColumnsToContents()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load CSV:\n{e}"
            )

    def save_csv(self):
        try:
            rows = self.table.rowCount()

            data = []

            for row in range(rows):
                name = self.table.item(row, 0).text()
                value = self.table.item(row, 1).text()
                unit = self.table.item(row, 2).text()

                data.append([name, value, unit])

            new_df = pd.DataFrame(
                data,
                columns=["name", "value", "unit"]
            )

            new_df.to_csv(CSV_FILE, index=False)

            QMessageBox.information(
                self,
                "Success",
                "Inputs saved successfully."
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save CSV:\n{e}"
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = InputEditor()
    window.show()
    app.exec()  # Blocks here until the user closes the input window
