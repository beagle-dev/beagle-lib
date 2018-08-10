package beagle;

/**
 * An interface for reporting information about the available benchmarked resources
 * as reported by the BEAGLE API.
 * @author Daniel Ayres
 * (based on ResourceDetails.java by Andrew Rambaut)
 * @version $Id$
 */
public class BenchmarkedResourceDetails {

    public BenchmarkedResourceDetails(int number) {
        this.number = number;
    }

    public int getNumber() {
        return number;
    }

    public int getResourceNumber() {
        return resourceNumber;
    }

    public void setResourceNumber(int resourceNumber) {
        this.resourceNumber = resourceNumber;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public long getSupportFlags() {
        return supportFlags;
    }

    public void setSupportFlags(long supportFlags) {
        this.supportFlags = supportFlags;
    }

    public long getRequiredFlags() {
        return requiredFlags;
    }

    public void setRequiredFlags(long requiredFlags) {
        this.requiredFlags = requiredFlags;
    }

    public int getReturnCode() {
        return returnCode;
    }

    public void setReturnCode(int returnCode) {
        this.returnCode = returnCode;
    }

    public String getImplName() {
        return implName;
    }

    public void setImplName(String implName) {
        this.implName = implName;
    }

    public long getBenchedFlags() {
        return benchedFlags;
    }

    public void setBenchedFlags(long benchedFlags) {
        this.benchedFlags = benchedFlags;
    }

    public double getBenchmarkResult() {
        return benchmarkResult;
    }

    public void setBenchmarkResult(double benchmarkResult) {
        this.benchmarkResult = benchmarkResult;
    }

    public double getPerformanceRatio() {
        return performanceRatio;
    }

    public void setPerformanceRatio(double performanceRatio) {
        this.performanceRatio = performanceRatio;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("  Resource ").append(getResourceNumber()).append(": ").append(getName()).append("\n");
        if (getDescription() != null) {
            String[] description = getDescription().split("\\|");
            for (String desc : description) {
                if (desc.trim().length() > 0) {
                    sb.append("    ").append(desc.trim()).append("\n");
                }
            }
        }
        sb.append("    Benchmark Flags:");
        sb.append(BeagleFlag.toString(getBenchedFlags()));
        sb.append("\n");
        sb.append("    Benchmark Result: ");
        sb.append(String.format ("%.3f", getBenchmarkResult()));
        sb.append(" ms (");
        sb.append(String.format ("%.2f", getPerformanceRatio()));
        sb.append("x CPU)");
        sb.append("\n");
        return sb.toString();
    }

    private final int number;
    private int resourceNumber;
    private String name;
    private String description;
    private long supportFlags;
    private long requiredFlags;
    private int returnCode;
    private String implName;
    private long benchedFlags;
    private double benchmarkResult;
    private double performanceRatio;
}
