import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import requests, os
from gwpy.timeseries import TimeSeries
from gwosc.locate import get_urls
from gwosc import datasets

# -- Default detector list
detectorlist = ['H1','L1', 'V1']

# Title the app
st.title('Signal Processing Tutorial')

st.markdown("## Make a signal with 3 sine waves")

cropstart = 1.0
cropend   = 1.3

def makesine(freq, amp):
    fs = 4096
    time = np.arange(0,3, 1.0/fs)
    y1 = amp*np.sin( 2*np.pi*freq*time )
    sig1 = TimeSeries(y1, dt=1.0/fs).taper() # ALS: Effect visible in plot: need to address or hide.
    plt.figure()
    fig_sig1 = sig1.crop(cropstart, cropend).plot()
    plt.xlim(cropstart, cropend)
    plt.ylim(-5,5)
    plt.title('Frequency {0} Hz - Amplitude {1}'.format(freq,amp))
    st.pyplot(fig_sig1, clear_figure=True)
    return(sig1)


st.sidebar.markdown("Controls for sine wave 1")
st.markdown("### Sine Wave 1")
freq1 = st.sidebar.slider("Frequency", 20, 200, 20)
amp1 = st.sidebar.slider("Amplitude", 1.0, 5.0, 5.0)
sig1 = makesine(freq1, amp1)


st.sidebar.markdown("Controls for sine wave 2")
st.markdown("### Sine Wave 2")
freq2 = st.sidebar.slider("Frequency", 20, 200, 103)
amp2 = st.sidebar.slider("Amplitude", 1.0, 5.0, 2.0)
sig2 = makesine(freq2, amp2)


st.sidebar.markdown("Controls for sine wave 3")
st.markdown("### Sine Wave 3")
freq3 = st.sidebar.slider("Frequency", 20, 200, 195)
amp3 = st.sidebar.slider("Amplitude", 1.0, 5.0, 4.0)
sig3 = makesine(freq3, amp3)

st.markdown("## Add the 3 sine waves together")
plt.figure()
signal = sig1 + sig2 + sig3
figsum = signal.crop(cropstart, cropend).plot()
plt.xlim(cropstart, cropend)
plt.title("Total signal in time domain")
st.pyplot(figsum, clear_figure=True)


st.markdown("## Convert to the frequency domain")


freqdomain = signal.fft()
# sigfig = np.abs(freqdomain).plot()
plt.figure()
plt.plot(freqdomain.frequencies, np.abs(freqdomain))
plt.title("Total signal in frequency domain")
plt.ylim(0,5)
plt.xlim(0,250)
st.pyplot(clear_figure=True)
#st.pyplot(sigfig)


st.markdown("## Try a band-pass filter")


lowfreq = st.slider("Low frequency cut-off", 1, 49, 5)
highfreq = st.slider("High frequency cut-off", 50, 300, 150)
bp_data = signal.bandpass(lowfreq, highfreq)

plt.figure()
bpfig = bp_data.crop(cropstart, cropend).plot()
st.pyplot(bpfig, clear_figure=True)

freqdomain = bp_data.fft()
plt.figure()
plt.plot(freqdomain.frequencies, np.abs(freqdomain))
plt.title("Band-passed signal in frequency domain")
plt.ylim(0,5)
plt.xlim(0,250)
st.pyplot()

# -- log plotting
#plt.figure()
#bpfig = np.abs(freqdomain).plot()
#plt.ylim(0,5)




st.markdown("## Try Whitening")

# white = signal.whiten()
white = sig1 / amp1 + sig2 / amp2 + sig3/amp3 # ALS: Why not use the whiten method?
plt.figure()
whitefig = white.plot()
plt.xlim(0,0.3)
st.pyplot(whitefig)

plt.figure()
freqdomain = white.fft()
plt.figure()
plt.plot(freqdomain.frequencies, np.abs(freqdomain))
plt.title("Whitened signal in frequency domain")
plt.ylim(0,5)
plt.xlim(0,250)
st.pyplot()

#whitefig = np.abs(freqdomain).plot()
#plt.ylim(0,5)
#st.pyplot(whitefig)

# -- Close all open figures
plt.close('all')


st.markdown("# Now try it with real data!")

@st.cache   #-- Magic command to cache data
def load_gw(t0, detector):
    strain = TimeSeries.fetch_open_data(detector, t0-18, t0+18, cache=False)
    return strain

part2 = st.checkbox('Ready to see it on real data?', value=False)

if part2:

    detector = 'H1'
    t0 = 1126259462.4   #-- GW150914

    st.text("Detector: {0}".format(detector))
    st.text("Time: {0} (GW150914)".format(t0))
    strain = load_gw(t0, detector)
    center = int(t0)
    strain = strain.crop(center-16, center+16)

    #-- Make a time series plot    

    st.subheader('Raw data')
    plt.figure()
    fig1 = strain.plot()
    plt.xlim(t0-0.2, t0+0.1)
    st.pyplot(fig1)

    # -- Plot asd
    plt.figure()
    psdfig = strain.asd(fftlength=4).plot()
    plt.xlim(10, 1800)
    st.pyplot(psdfig)

    # -- Try whitened and band-passed plot
    # -- Whiten and bandpass data
    st.subheader('Whitened and Bandbassed Data')

    lowfreqreal = st.slider("Low frequency cut-off", 1, 49, 40)
    highfreqreal = st.slider("High frequency cut-off", 50, 1000, 300)
    makewhite = st.checkbox("Apply whitening", value=True)

    if makewhite:
        white_data = strain.whiten()
    else:
        white_data = strain

    bp_data = white_data.bandpass(lowfreqreal, highfreqreal)
    fig3 = bp_data.plot()
    plt.xlim(t0-0.2, t0+0.1)
    st.pyplot(fig3)


    # -- PSD of whitened data
    # -- Plot psd
    plt.figure()
    psdfig = bp_data.asd(fftlength=4).plot()
    plt.xlim(10, 1800)
    st.pyplot(psdfig)


    # -- Close all open figures
    plt.close('all')
