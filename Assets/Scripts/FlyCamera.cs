
using UnityEngine;
using UnityEngine.EventSystems;
using System.Collections;

namespace Nezix {
public class FlyCamera : MonoBehaviour {

    public float speed = 0.1f;
    public float speedLook = 0.5f;
    Vector2 rotation = new Vector2 (0, 0);

    void Update () {
        GameObject curUIInput = EventSystem.current.currentSelectedGameObject;

        if (curUIInput == null && Input.GetKey (KeyCode.W)) {
            transform.Translate(transform.forward * speed);
        }
        if (curUIInput == null && Input.GetKey (KeyCode.S)) {
            transform.Translate(-transform.forward * speed);
        }
        if (curUIInput == null && Input.GetKey (KeyCode.Q)) {
            transform.Translate(-transform.right * speed);
        }
        if (curUIInput == null && Input.GetKey (KeyCode.D)) {
            transform.Translate(transform.right * speed);
        }
        if (curUIInput == null && Input.GetKey (KeyCode.Space)) {
            transform.Translate(transform.up * speed);
        }
        if (curUIInput == null && Input.GetKey (KeyCode.Z)) {
            transform.Translate(-transform.up * speed);
        }
        if (Input.GetMouseButton (0) && (Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.RightControl))) {
            rotation.y += Input.GetAxis ("Mouse X");
            rotation.x += -Input.GetAxis ("Mouse Y");
            transform.eulerAngles = (Vector2)rotation * speedLook;
        }
    }
}
}