const captionForm = document.querySelector(".js-cap"),
  
      resultcaption= document.querySelector(".resultcap"),

      captionBtn = document.querySelector("#captionBtn");




const USER_LS = "currentUser";
const SHOWING_CN = "showing";

function askForcap(event) {
    event.preventDefault();
    captionForm.classList.add(SHOWING_CN);
    resultcaption.classList.remove(SHOWING_CN);

}






function init() {

    captionBtn.addEventListener("click", askForcap);

}

init();