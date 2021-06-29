import pyautogui
from pyclick.humancurve import HumanCurve
##--##
import time
import pandas as pd

MAX_RES = [1920, 1080]

def setup_pyautogui():
    # Any duration less than this is rounded to 0.0 to instantly move the mouse.
    pyautogui.MINIMUM_DURATION = 0  # Default: 0.1
    # Minimal number of seconds to sleep between mouse moves.
    pyautogui.MINIMUM_SLEEP = 0  # Default: 0.05
    # The number of seconds to pause after EVERY public function call.
    pyautogui.PAUSE = 0.015  # Default: 0.1

setup_pyautogui()

class HumanClicker():
    def __init__(self):
        # pass
        ##--##
        self.humanCurve = None

    def move(self, toPoint, duration=2, humanCurve=None):
        fromPoint = pyautogui.position()
        if not self.humanCurve:
            self.humanCurve = HumanCurve(fromPoint, toPoint)

        pyautogui.PAUSE = duration / len(self.humanCurve.points)
        for point in self.humanCurve.points:
            print(point)
            pyautogui.moveTo(point)

    def click(self):
        pyautogui.click()

    ##--## Extend base class with more functions

    def getPoints(self, fromPoint, toPoint, size):
        '''
            Generate Beziere curve coordinates
            betwen fromPoint and toPoint,
            in a specified size
        '''
        self.humanCurve = HumanCurve(fromPoint, toPoint, targetPoints = size)
        return self.humanCurve.points

    # def generateAction(self, toPoint, duration, sizePoints):
    #     fromPoint = pyautogui.position()
        
    #     self.humanCurve = HumanCurve(fromPoint, toPoint, targetPoints = sizePoints)

    #     action_df = pd.DataFrame({})
        
    #     pyautogui.PAUSE = duration / len(self.humanCurve.points)
    #     for point in self.humanCurve.points:
    #         ## resolution check
    #         if point[0] > (MAX_RES[0] - 30):
    #             point = (point[0] - 30, point[1])
            
    #         if point[1] > (MAX_RES[1] - 15):
    #             point = (point[0], point[1] - 15)

    #         pyautogui.moveTo(point)
    #         timestamp = round(time.time(), 3) * 1000    # miliseconds
    #         action_df = action_df.append(pd.DataFrame([[timestamp, point[0], point[1]]], columns=['client timestamp', 'x', 'y']), ignore_index=True)

    #     return action_df

    def getMousePosition(c):
        return pyautogui.position()
