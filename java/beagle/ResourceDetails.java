package beagle;

/**
 * An interface for reporting information about the available resources
 * as reported by the BEAGLE API.
 * @author Andrew Rambaut
 * @version $Id$
 */
public class ResourceDetails {

    public ResourceDetails(int number) {
        this.number = number;
    }

    public int getNumber() {
        return number;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getDetails() {
        return details;
    }

    public void setDetails(String details) {
        this.details = details;
    }

    public long getFlags() {
        return flags;
    }

    public void setFlags(long flags) {
        this.flags = flags;
    }

    private final int number;
    private String name;
    private String details;
    private long flags;
}
