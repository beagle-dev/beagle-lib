package beagle;

/**
 * @author Andrew Rambaut
 * @author Alexei Drummond
 * @version $Id$
 */
public enum BeagleFlag {
    DOUBLE(1 << 0, "Request/require double precision computation"),
    SINGLE(1 << 1, "Request/require single precision computation"),
    ASYNCH(1 << 2, "Request/require asynchronous computation"),
    SYNCH(1 << 3, "Request/require synchronous computation"),
    CPU(1 << 16, "Request/require CPU"),
    GPU(1 << 17, "Request/require GPU"),
    FPGA(1 << 18, "Request/require FPGA"),
    SSE(1 << 19, "Request/require SSE"),
    CELL(1 << 20, "Request/require Cell");

    BeagleFlag(long mask, String meaning) {
        this.mask = mask;
        this.meaning = meaning;
    }

    public long getMask() {
        return mask;
    }

    public String getMeaning() {
        return meaning;
    }

    public boolean isSet(long flags) {
        return (flags & mask) != 0;
    }

    private final long mask;
    private final String meaning;
}