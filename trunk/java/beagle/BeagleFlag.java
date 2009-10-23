package beagle;

/**
 * @author Andrew Rambaut
 * @author Marc Suchard
 * @version $Id$
 */
public enum BeagleFlag {
    DOUBLE(1 << 0, "Request/require double precision computation"),
    SINGLE(1 << 1, "Request/require single precision computation"),
    ASYNCH(1 << 2, "Request/require asynchronous computation"),
    SYNCH(1 << 3, "Request/require synchronous computation"),
    COMPLEX(1 <<4, "Request/require complex diagonalization capability"),
    LSCALE(1 << 5, "Request/require storing scalars on log-scale"),
    CPU(1 << 16, "Request/require CPU"),
    GPU(1 << 17, "Request/require GPU"),
    FPGA(1 << 18, "Request/require FPGA"),
    SSE(1 << 19, "Request/require SSE"),
    CELL(1 << 20, "Request/require Cell"),
    OPENMP(1 << 21, "Request/require OpenMP");

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

    public static String toString(long flags) {
        StringBuilder sb = new StringBuilder();
        for (BeagleFlag flag : BeagleFlag.values()) {
            if (flag.isSet(flags)) {
                sb.append(" ").append(flag.name());
            }
        }
        return sb.toString();
    }

    private final long mask;
    private final String meaning;
}