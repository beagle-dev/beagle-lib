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
            System.out.println("\t" + resource.getNumber() + " : " + resource.getName() + " [" + resource.getFlags() + "]");
        }

    }

}
