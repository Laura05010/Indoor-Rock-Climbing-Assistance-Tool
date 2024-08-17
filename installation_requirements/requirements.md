## Recommended Setup

It is recommended that you use an Anaconda environment to run this project. By doing so, you can avoid conflicts with preinstalled libraries on your computer.

### Setting Up a Conda Environment

1. **Create a new Conda environment:**
   ```bash
   conda create --name myenv python=3.9
Replace myenv with your preferred environment name

2. **Activate the Conda environment:**
   ```bash
    conda activate myenv

3. **Install required dependencies by running the bash file:**
   ```bash
    ./installation_requirements/install_dependencies.sh

4. **Navigate to the directory where main.py is located:**
    ```bash
    cd path/to/your/directory

5. **Run the application:**
    ```bash
    python main.py

6. **Deactivate the Conda environment when you're done running the code:**
    ```bash
    conda deactivate


### Important for Mac Users

Please make sure that you have accessibility permissions for the IDE where you are running the code, otherwise, you will not be able to listen for user input. To do this:

1. Open `System Preferences` -> `Security & Privacy`.
2. Click on the `Privacy` tab.
3. Scroll and click on the `Accessibility` row.
4. Click the `+` button.
5. Navigate to `/System/Applications/Utilities/` or wherever the `Terminal.app` is installed.
6. Click `Open`.

If you are using an IDE, you also need to:

1. Open `System Settings` -> `Privacy & Security`.
2. Open `Input Monitoring`.
3. Press the `+` button and add the IDE app that you are using to run the code to the list.

Reference: [Stackflow Discussion](https://stackoverflow.com/questions/69620702/this-process-is-not-trusted-when-running-code-in-pycharm)