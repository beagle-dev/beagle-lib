package beagle;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Marc A. Suchard
 * @author Andrew Rambaut
 */
public class BeagleInfo {

    public static String getVersion() {
        return BeagleFactory.getVersion();
    }

    public static void printVersionInformation() {

        System.out.println(BeagleFactory.getVersionInformation());
        System.out.println();
    }

    public static void printResourceList() {

        List<ResourceDetails> resourceDetails = BeagleFactory.getResourceDetails();

        System.out.println("BEAGLE resources available:");
        for (ResourceDetails resource : resourceDetails) {
            System.out.println(resource.toString());
            System.out.println();

        }
    }

    public static void main(String[] argv) {
        printVersionInformation();
        printResourceList();
    }

}
