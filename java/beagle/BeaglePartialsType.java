package beagle;

/**
 * @author Marc Suchard
 * @author Filippo Monti
 * @version $Id$
 */
public enum BeaglePartialsType {
	BOTTOM(1 << 0, "bottom"),
	TOP(1 << 1, "top");    

    BeaglePartialsType(int type, String meaning) {
        this.type = type;
        this.meaning = meaning;
    }

    public int getType() {
        return type;
    }

    public String getMeaning() {
        return meaning;
    }

    private final int type;
    private final String meaning;
}

