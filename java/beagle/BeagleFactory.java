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

    public static Beagle loadBeagleInstance(
            int tipCount,
            int partialsBufferCount,
            int compactBufferCount,
            int stateCount,
            int patternCount,
            int eigenBufferCount,
            int matrixBufferCount,
            int categoryCount
    ) {

        boolean forceJava = Boolean.valueOf(System.getProperty("java_only"));

        if (BeagleJNIWrapper.INSTANCE == null) {
            try {
                BeagleJNIWrapper.loadBeagleLibrary();
                System.err.println("BEAGLE library loaded");

            } catch (UnsatisfiedLinkError ule) {
                System.err.println("Failed to load BEAGLE library");
            }
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
                    categoryCount,
                    null,
                    0,
                    0
            );
        }

//        if (stateCount == 4) {
////            return new DependencyAwareBeagleImpl();
//            return new FourStateBeagleImpl(
//                    tipCount,
//                    partialsBufferCount,
//                    compactBufferCount,
//                    stateCount,
//                    patternCount,
//                    eigenBufferCount,
//                    matrixBufferCount
//            );
//        }
        return new GeneralBeagleImpl(tipCount,
                partialsBufferCount,
                compactBufferCount,
                stateCount,
                patternCount,
                eigenBufferCount,
                matrixBufferCount,
                categoryCount
        );
    }

    public static void main(String[] argv) {
        Beagle instance = BeagleFactory.loadBeagleInstance(3, 2, 3, 4, 1, 5, 5, 1);
    }

}
