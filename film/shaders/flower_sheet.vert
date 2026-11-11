#version 410
// the petal from straight above the drop: its height, and which part of it this is
in vec3 in_p; in vec2 in_uv;
uniform vec3 uO;
uniform float uRd,uL,uY0,uY1;
out vec3 vQ;
void main(){
  vec3 q=(in_p-uO)/uRd;
  vQ=vec3(q.y,in_uv);
  gl_Position=vec4(q.x/uL,q.z/uL,(q.y-uY0)/(uY1-uY0)*2.-1.,1.);
}
