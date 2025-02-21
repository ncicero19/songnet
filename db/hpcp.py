## The (slightly modified) code from Essentia to compute HPCP chromagrams (an arguably better substitute for tonnetz)##
## This is the backbone of tonnetz.py, which is just a pipeline to pull and store data##

import essentia.streaming as ess
import essentia

def hpcp(audio_file):
    # Initialize algorithms
    loader = ess.MonoLoader(filename=audio_file)
    framecutter = ess.FrameCutter(frameSize=1024, hopSize=512, silentFrames='noise')
    windowing = ess.Windowing(type='blackmanharris62')
    spectrum = ess.Spectrum()
    spectralpeaks = ess.SpectralPeaks(orderBy='magnitude',
                                      magnitudeThreshold=0.00001,
                                      minFrequency=20,
                                      maxFrequency=3500,
                                      maxPeaks=60)
    
    hpcp = ess.HPCP()

    # Use pool to store data
    pool = essentia.Pool()

    # Connect streaming algorithms
    loader.audio >> framecutter.signal
    framecutter.frame >> windowing.frame >> spectrum.frame
    spectrum.spectrum >> spectralpeaks.spectrum
    spectralpeaks.magnitudes >> hpcp.magnitudes
    spectralpeaks.frequencies >> hpcp.frequencies
    hpcp.hpcp >> (pool, 'tonal.hpcp')

    # Run processing
    essentia.run(loader)

    # Convert to a list (for easier debugging and visualization)
    return pool['tonal.hpcp']
