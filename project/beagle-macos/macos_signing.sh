#!/bin/bash

rm -rf full_expand
rm -f intermediate.pkg

pkgutil --expand-full ${INITIAL_FILE_NAME} full_expand

find full_expand -type f -name "*.so" -print0 | while IFS= read -r -d $'\0' FILE; do
  echo "Signing: ${FILE}"
  OUTPUT=$(codesign -s "${APPLE_APPLICATION_IDENTITY_SHA}" --force --options runtime ${FILE} 2>&1)
  RESULT=$?
  
  if [ "${RESULT}" -eq 0 ]; then
  	echo "Signed: ${FILE}"
  else
    echo "Error: Failed to sign ${FILE}: ${OUTPUT}" >&2
    exit 1
  fi
done

find full_expand -type f -name "*.dylib" -print0 | while IFS= read -r -d $'\0' FILE; do
  echo "Signing: ${FILE}"
  OUTPUT=$(codesign -s "${APPLE_APPLICATION_IDENTITY_SHA}" --force --options runtime ${FILE} 2>&1)
  RESULT=$?
  
  if [ "${RESULT}" -eq 0 ]; then
  	echo "Signed: ${FILE}"
  else
    echo "Error: Failed to sign ${FILE}: ${OUTPUT}" >&2
    exit 1
  fi
done

pkgutil --flatten full_expand intermediate.pkg

productsign --sign "${APPLE_INSTALLER_IDENTITY_SHA}" intermediate.pkg ${FINAL_FILE_NAME}

REQUEST_UUID=$(xcrun notarytool submit ${FINAL_FILE_NAME} --keychain-profile "${KEYCHAIN_NAME}" --wait | grep "id:" | head -1 | awk '{print $2}')

if [[ -n "$REQUEST_UUID" ]]; then
  echo "Notarization submitted with UUID: $REQUEST_UUID"
  xcrun notarytool log "$REQUEST_UUID" --keychain-profile "${KEYCHAIN_NAME}"
  xcrun stapler staple ${FINAL_FILE_NAME}
  echo "Notarization successful and stapled."
else
  echo "Notarization submission failed."
  exit 1
fi          
