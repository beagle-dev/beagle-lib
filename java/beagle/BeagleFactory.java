/**
 *
 */
package beagle;

import java.util.*;

/**
 * @author Marc Suchard
 * @author Andrew Rambaut
 *
 */
public class BeagleFactory {

    public static final String STATE_COUNT = "stateCount";
    public static final String PREFER_SINGLE_PRECISION = "preferS inglePrecision";
    public static final String SINGLE_PRECISION = "singlePrecision";
    public static final String DEVICE_NUMBER = "deviceNumber";

    private static Beagle load(BeagleLoader loader, Map<String, Object> configuration) {
        // configuration.put(BeagleFactory.SINGLE_PRECISION, singlePrecision);

        System.out.print("Attempting to load beagle: " + loader.getLibraryName(configuration));

        Beagle beagle = loader.createInstance(configuration);

        if (beagle != null) {
            System.out.println(" - SUCCESS");

            return beagle;
        }
        System.out.println(" - FAILED");
        return null;
    }

    public static Beagle loadBeagleInstance(Map<String, Object> configuration) {

        if (registry == null) {  // Lazy loading
            registry = new ArrayList<BeagleLoader>();  // List libraries in order of load-priority
            registry.add(new BeagleJNIWrapper.BeagleLoader());
        }

        for(BeagleLoader loader: registry) {

            // Try prefered precision library
            Beagle beagle = load(loader, configuration);
            if (beagle != null)
                return beagle;
        }

        // No libraries/processes available

        int stateCount = (Integer)configuration.get(STATE_COUNT);
//        boolean singlePrecision = (Boolean)configuration.get(PREFER_SINGLE_PRECISION);
//
//        if (singlePrecision) {
//            // return new SinglePrecisionBeagleImpl(stateCount);
//            // throw new UnsupportedOperationException("Single precision Java version of BEAGLE not implemented");
//            System.out.println("Single precision Java version of BEAGLE not available; defaulting to double precision");
//        } // else {

        if (stateCount == 4) {
            return new FourStateBeagleImpl();
        }
        return new GeneralBeagleImpl(stateCount);
    }

    private static List<BeagleLoader> registry;

    protected interface BeagleLoader {
        public String getLibraryName(Map<String, Object> configuration);

        /**
         * Actual factory
         * @param configuration
         * @return
         */
        public Beagle createInstance(Map<String, Object> configuration);
    }

    public static void main(String[] argv) {
        Map<String, Object> configuration = new HashMap<String, Object>();
        configuration.put(BeagleFactory.STATE_COUNT, 4);
        configuration.put(BeagleFactory.PREFER_SINGLE_PRECISION, false);
        configuration.put(BeagleFactory.DEVICE_NUMBER, 0);

        Beagle instance = BeagleFactory.loadBeagleInstance(configuration);
    }

}
