package beagle;

/**
 * @author Daniel Ayres
 * (based on BeagleFlag.java by Andrew Rambaut and Marc Suchard)
 * @version $Id$
 */
public enum BeagleBenchmarkFlag {
    SCALING_NONE(1 << 0, "No scaling"),
    SCALING_ALWAYS(1 << 1, "Scale at every iteration"),
    SCALING_DYNAMIC(1 << 2, "Scale every fixed number of iterations or when a numerical error occurs, and re-use scale factors for subsequent iterations");

    BeagleBenchmarkFlag(long mask, String meaning) {
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
        for (BeagleBenchmarkFlag flag : BeagleBenchmarkFlag.values()) {
            if (flag.isSet(flags)) {
                sb.append(" ").append(flag.name());
            }
        }
        return sb.toString();
    }

    private final long mask;
    private final String meaning;
}