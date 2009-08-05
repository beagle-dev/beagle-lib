package beagle;

import java.util.ArrayList;

/**
 * @author Marc A. Suchard
 * @author Andrew Rambaut
 */
public class BeagleInfo {

    public static ResourceDetails[] getResourceDetails() {
        if (BeagleJNIWrapper.INSTANCE == null) {
            try {
                BeagleJNIWrapper.loadBeagleLibrary();
                System.err.println("BEAGLE library loaded");

            } catch (UnsatisfiedLinkError ule) {
                System.err.println("Failed to load BEAGLE library: " + ule.getMessage());
            }
        }

        return BeagleJNIWrapper.INSTANCE.getResourceList();
    }

    public static void main(String[] argv) {
        ResourceDetails[] resourceDetails = getResourceDetails();

        System.out.println("BEAGLE resources available:");
        for (ResourceDetails resource : resourceDetails) {
            System.out.println("\t" + resource.getNumber() + " : " + resource.getName());
            String[] description = resource.getDescription().split("\\|");
            for (String desc : description) {
                if (desc.trim().length() > 0) {
                    System.out.println("\t\t" +  desc.trim());
                }
            }
            StringBuilder sb = new StringBuilder();
            for (BeagleFlag flag : BeagleFlag.values()) {
                if (flag.isSet(resource.getFlags())) {
                    sb.append(" ").append(flag.name());
                }
            }
            System.out.println("\t\tFlags:" + sb.toString());
            System.out.println();

        }

    }

}
