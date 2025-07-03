#!/Users/ssg/.pyenv/shims/python
# -*- coding: utf-8 -*-
import sys
import time
import datetime
import argparse
from os import listdir
from os.path import split, splitext, isfile
from parameterSetup import ParameterSetup
from classifierClient import ClassifierClient
from eegFileReaderServer import EEGFileReaderServer
from fileManagement import selectClassifierID

class RemOfflineApplication:

    def __init__(self, args):
        self.args = args

    def start(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('--samplingFreq', type=int, default=128, help='set sampling frequency')
        parser.add_argument('--windowSizeInSec', type=int, default=4, help='set size of window')
        parser.add_argument('--stepSizeInSec', type=int, default=1, help='set size of sliding-step')
        parser.add_argument('--postdir', type=str, default="../data/aipost", help='Path to EEGdata(postDir)')
        parser.add_argument('--classifier_id', type=str, default=None,help='如果指定,就强制使用这个训练好的模型ID,跳过自动匹配逻辑')
        parser.add_argument('--output_the_same_fileID', action='store_true', help='Force using fixed file ID for output')
        args_parsed = parser.parse_args(self.args[1:])

        params = ParameterSetup()

        self.recordWaves = params.writeWholeWaves
        self.extractorType = params.extractorType
        self.classifierType = params.classifierType
        self.postDir = args_parsed.postdir if args_parsed.postdir else params.postDir
        self.predDir = params.predDir
        self.finalClassifierDir = params.finalClassifierDir

        observed_samplingFreq = args_parsed.samplingFreq
        observed_epochTime = args_parsed.windowSizeInSec
        stepSizeInSec = args_parsed.stepSizeInSec
        postDir = args_parsed.postdir
        useFixedID = args_parsed.output_the_same_fileID
        forced_id = args_parsed.classifier_id
        # eegFilePath = args[1]
        # inputFileID = splitext(split(eegFilePath)[1])[0]
        postFiles = listdir(self.postDir)
        fileCnt = 0
        for inputFileName in postFiles:
            if not inputFileName.startswith('.'):
                print('inputFileName = ' + inputFileName)
                inputFileID = splitext(inputFileName)[0]
                print('inputFileID = ' + inputFileID)
                predFileFullPath = self.predDir + '/' + inputFileID + '_pred.txt'
                print('predFileFullPath = ' + predFileFullPath)

                if not isfile(predFileFullPath):
                    fileCnt += 1
                    print('  processing ' + inputFileID)
                    print("Requested classifierType =", self.classifierType)
                    try:
                        
                        classifierID, model_samplingFreq, model_epochTime = selectClassifierID(self.finalClassifierDir, self.classifierType, requested_samplingFreq=observed_samplingFreq, requested_epochTime=observed_epochTime,
                                                                                               forced_classifierID=forced_id)
                        if useFixedID:
                            self.client = ClassifierClient(self.recordWaves, self.extractorType, self.classifierType, classifierID="W98DEW", inputFileID=inputFileID,
                                                                samplingFreq=model_samplingFreq, epochTime=model_epochTime, stepSizeInSec=stepSizeInSec)
                        else:
                            self.client = ClassifierClient(self.recordWaves, self.extractorType, self.classifierType, classifierID,
                                samplingFreq=model_samplingFreq, epochTime=model_epochTime, stepSizeInSec=stepSizeInSec)
                        self.client.predictionStateOn()
                        self.client.hasGUI = False
                        # sys.stdout.write('classifierClient started by ' + str(channelOpt) + ' channel.')

                    except Exception as e:
                        print(str(e))
                        raise e

                    try:
                        eegFilePath = self.postDir + '/' + inputFileName
                        self.server = EEGFileReaderServer(self.client, eegFilePath, model_samplingFreq=model_samplingFreq, model_epochTime=model_epochTime,
                            observed_samplingFreq=observed_samplingFreq, observed_epochTime=observed_epochTime)

                    except Exception as e:
                        print(str(e))
                        raise e

                else:
                    print('  skipping ' + inputFileID + ' because ' + predFileFullPath + ' exists.')


if __name__ == '__main__':
    start_time = time.time()
    args = sys.argv
    mainapp = RemOfflineApplication(args)
    mainapp.start()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[INFO] Total runtime: {str(datetime.timedelta(seconds=int(elapsed)))}")
    # while True:
        # print('*')
        # time.sleep(5)
    # sys.exit(app.exec_())
