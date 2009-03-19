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
	public static final String SINGLE_PRECISION = "singlePrecision";
    public static final String INSTANCE_NUMBER = "instanceNumber";

	public static Beagle loadBeagleInstance(Map<String, Object> configuration) {

		if (registry == null) {  // Lazy loading
			registry = new ArrayList<BeagleLoader>();  // List libraries in order of load-priority
            registry.add(new BeagleJNIWrapper.BeagleLoader());
		}

		for(BeagleLoader loader: registry) {
            System.out.print("Attempting to load beagle: " + loader.getLibraryName(configuration));

            Beagle beagle = loader.createInstance(configuration);

			if (beagle != null) {
                System.out.println(" - SUCCESS");

				return beagle;
            }
            System.out.println(" - FAILED");
		}

		// No libraries/processes available

		int stateCount = (Integer)configuration.get(STATE_COUNT);
		boolean singlePrecision = (Boolean)configuration.get(SINGLE_PRECISION);

		if (singlePrecision) {
			// return new SinglePrecisionBeagleImpl(stateCount);
            throw new UnsupportedOperationException("Single precision Java version of BEAGLE not implemented");
        } else {
			return new DoublePrecisionBeagleImpl(stateCount);
        }
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
        configuration.put(BeagleFactory.SINGLE_PRECISION, false);
        configuration.put(BeagleFactory.INSTANCE_NUMBER, 0);

        Beagle instance = BeagleFactory.loadBeagleInstance(configuration);
    }

}
