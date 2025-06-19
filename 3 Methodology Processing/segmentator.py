
#fs: Sampling Rate in Hz
#tseg = 0.100  # segment duration (s)

def segmentator(signal,fs, tseg,toverlap,N):
    # Prepare to segment the data
    import numpy as np
    nseg = int(tseg * fs)  # segment length (points)
    noverlap = int(toverlap * nseg)  # overlap (points)
    noffset = nseg - noverlap  # offset (points)
    K = int(np.floor(1 + (N - nseg) / noffset))  # number of segments
    yseg = np.zeros((K, nseg))  # allocate arrays for time and y for each segment
    for i in range(1,K,1):
        yseg[i, :] = signal[ (i - 1) * noffset: (i - 1) * noffset + nseg]
    return yseg