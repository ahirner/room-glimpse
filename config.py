#Config Camera
RESOLUTION = (640, 480)
FPS = 30
ROTATION = 180
MD_BLOCK_FRACTION = 0.008 #Fraction of blocks that must show movement
MD_SPEED = 2.0            #How many screens those blocks must move per second
MD_FALLOFF = 0.75         #How many seconds no motion must be present to trigger completion of a scene

#Config Persistency
DATA_FOLDER = './data'

#Config Azure Cognition
AZURE_COG_HOST = 'https://westus.api.cognitive.microsoft.com/vision/v1.0/analyze'
AZURE_COG_RETRIES = 3