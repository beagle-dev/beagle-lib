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
    public static final String PREFER_SINGLE_PRECISION = "preferSinglePrecision";
    public static final String SINGLE_PRECISION = "singlePrecision";
    public static final String DEVICE_NUMBER = "deviceNumber";

    public static Beagle loadBeagleInstance(
            int tipCount,
            int partialsBufferCount,
            int compactBufferCount,
            int stateCount,
            int patternCount,
            int eigenBufferCount,
            int matrixBufferCount) {

        boolean forceJava = Boolean.valueOf(System.getProperty("java_only"));

        if (BeagleJNIWrapper.INSTANCE == null) {
            BeagleJNIWrapper.loadBeagleLibrary();
        }

        if (!forceJava && BeagleJNIWrapper.INSTANCE != null) {
            return new BeagleJNIImpl(
                    tipCount,
                    partialsBufferCount,
                    compactBufferCount,
                    stateCount,
                    patternCount,
                    eigenBufferCount,
                    matrixBufferCount,
                    null,
                    0,
                    0
            );
        }


        // No libraries/processes available

        if (stateCount == 4) {
//            return new DependencyAwareBeagleImpl();
            return new FourStateBeagleImpl(
                    tipCount,
                    partialsBufferCount,
                    compactBufferCount,
                    stateCount,
                    patternCount,
                    eigenBufferCount,
                    matrixBufferCount
            );
        }
        return new DefaultBeagleImpl(tipCount,
                    partialsBufferCount,
                    compactBufferCount,
                    stateCount,
                    patternCount,
                    eigenBufferCount,
                    matrixBufferCount);
    }

    public static void main(String[] argv) {
        Beagle instance = BeagleFactory.loadBeagleInstance(3, 2, 3, 4, 1, 5, 5);
    }

}
