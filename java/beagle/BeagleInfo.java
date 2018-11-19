package beagle;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author Marc A. Suchard
 * @author Andrew Rambaut
 */
public class BeagleInfo {

    public static String getVersion() {
        return BeagleFactory.getVersion();
    }

    public static String getVersionInformation() {
        return BeagleFactory.getVersionInformation();
    }

    public static int[] getVersionNumbers() {
        String version = BeagleFactory.getVersion();
        Pattern p = Pattern.compile("(\\d+)\\.(\\d+)\\.(\\d+).*");
        Matcher m = p.matcher(version);
        if (m.matches()) {
            return new int[] {Integer.parseInt(m.group(1)), Integer.parseInt(m.group(2)), Integer.parseInt(m.group(3))};
        }
        return new int[] {};
    }

    public static void printVersionInformation() {

        System.out.println(getVersionInformation());
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
