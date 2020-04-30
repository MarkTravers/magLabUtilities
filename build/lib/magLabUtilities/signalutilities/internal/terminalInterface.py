#!python3

def printMsg(msg, mode):
    if mode == 'verbose':
        print(msg)
    elif isinstance(mode, MP):
        mode.out(['DSP'], msg)

class MP:
    def __init__(self, initMessage, messageTypes):
        # Stores the names (as strings) of each active message type
        self.messageTypesDict = messageTypes

        # Prints a custom message at initialization
        print('\n' + initMessage)

        if self.messageTypesDict.keys() == []:
            raise Exception('Message Pipe: No message types specified in initializer.')
        elif 'All' in self.messageTypesDict.keys() and 'None' in self.messageTypesDict.keys():
            raise Exception('Message Pipe: Cannot have both "All" and "None" messages.')
        elif 'All' in self.messageTypesDict.keys():
            if self.messageTypesDict['All'] == 1:
                for messageType in self.messageTypesDict:
                    self.messageTypesDict[messageType] = 1
        elif 'None' in self.messageTypesDict.keys():
            if self.messageTypesDict['None'] == 1:
                for messageType in self.messageTypesDict:
                    self.messageTypesDict[messageType] = 0

        # Lists all configured message types and whether they are enabled
        print ('\n    Message Types:')
        for messageType in self.messageTypesDict:
            if self.messageTypesDict[messageType] == 1:
                print('        %s: Enabled ' % '{:13}'.format(messageType))
            else:
                print('        %s: Disabled ' % '{:13}'.format(messageType))
        print('')

    # Receives a message and decides whether to display it on the terminal
    def out(self, messageTypes, message):
        if self.messageTypesDict == {}:
            raise Exception('Message Pipe: Pipe is not initialized.')
        else:
            for messageType in messageTypes:
                if messageType in self.messageTypesDict.keys():
                    if self.messageTypesDict[messageType] == 1:
                        print('%s>> %s' % ('{:13}'.format(messageType), message))
                else:
                    raise Exception('Message Pipe: Message type "%s" not recognized.' % messageType)