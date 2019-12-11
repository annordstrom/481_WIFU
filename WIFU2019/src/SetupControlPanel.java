
import java.awt.Component;
import javax.swing.JOptionPane;

public class SetupControlPanel extends javax.swing.JFrame {
    
     
    static void displayDialog(start aThis) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    /**
     * Creates new form SetupControlPanel
     */
    public SetupControlPanel() {
        initComponents();
        setExtendedState(start.MAXIMIZED_BOTH);
        firstNameText.setText(start.userFirstName);
        lastNameText.setText(start.userLastName);
        streetText.setText(start.userStreet);
        cityText.setText(start.userCity);
        zipText.setText(start.userZip);
        jComboBox1.setSelectedItem(start.userState);
        if(start.userBool == false)
        {
            nextButton.setEnabled(false);
        }

    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        userInformationLabel = new javax.swing.JLabel();
        streetLabel = new javax.swing.JLabel();
        firstNameLabel = new javax.swing.JLabel();
        firstNameText = new javax.swing.JTextField();
        lastNameText = new javax.swing.JTextField();
        lastNameLabel1 = new javax.swing.JLabel();
        streetText = new javax.swing.JTextField();
        javax.swing.JLabel cityLabel = new javax.swing.JLabel();
        zipText = new javax.swing.JTextField();
        javax.swing.JLabel stateLabel = new javax.swing.JLabel();
        jComboBox1 = new javax.swing.JComboBox<String>();
        javax.swing.JLabel zipLabel = new javax.swing.JLabel();
        cityText = new javax.swing.JTextField();
        nextButton = new javax.swing.JButton();
        MainMenu = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        userInformationLabel.setFont(new java.awt.Font("Tahoma", 1, 36)); // NOI18N
        userInformationLabel.setText("User Information");

        streetLabel.setFont(new java.awt.Font("Tahoma", 0, 30)); // NOI18N
        streetLabel.setText("Street:");

        firstNameLabel.setFont(new java.awt.Font("Tahoma", 0, 30)); // NOI18N
        firstNameLabel.setText("First Name:");

        firstNameText.setFont(new java.awt.Font("Tahoma", 0, 30)); // NOI18N

        lastNameText.setFont(new java.awt.Font("Tahoma", 0, 30)); // NOI18N

        lastNameLabel1.setFont(new java.awt.Font("Tahoma", 0, 30)); // NOI18N
        lastNameLabel1.setText("Last Name:");

        streetText.setFont(new java.awt.Font("Tahoma", 0, 30)); // NOI18N

        cityLabel.setFont(new java.awt.Font("Tahoma", 0, 30)); // NOI18N
        cityLabel.setText("City:");

        zipText.setFont(new java.awt.Font("Tahoma", 0, 30)); // NOI18N
        zipText.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                zipTextActionPerformed(evt);
            }
        });

        stateLabel.setFont(new java.awt.Font("Tahoma", 0, 30)); // NOI18N
        stateLabel.setText("State:");

        jComboBox1.setFont(new java.awt.Font("Tahoma", 0, 30)); // NOI18N
        jComboBox1.setModel(new javax.swing.DefaultComboBoxModel(new String[] { "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY" }));
        jComboBox1.setSelectedItem(jComboBox1);
        jComboBox1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jComboBox1ActionPerformed(evt);
            }
        });

        zipLabel.setFont(new java.awt.Font("Tahoma", 0, 30)); // NOI18N
        zipLabel.setText("Zip Code:");

        cityText.setFont(new java.awt.Font("Tahoma", 0, 30)); // NOI18N

        nextButton.setFont(new java.awt.Font("Tahoma", 1, 30)); // NOI18N
        nextButton.setText("Next");
        nextButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                nextButtonActionPerformed(evt);
            }
        });

        MainMenu.setFont(new java.awt.Font("Tahoma", 1, 30)); // NOI18N
        MainMenu.setText("Main Menu");
        MainMenu.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                MainMenuActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                    .addGroup(layout.createSequentialGroup()
                        .addContainerGap(71, Short.MAX_VALUE)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(zipLabel)
                                .addGap(18, 18, 18)
                                .addComponent(zipText, javax.swing.GroupLayout.PREFERRED_SIZE, 211, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGroup(layout.createSequentialGroup()
                                .addGap(3, 3, 3)
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addGroup(layout.createSequentialGroup()
                                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                                            .addComponent(firstNameLabel)
                                            .addComponent(lastNameLabel1))
                                        .addGap(18, 18, 18)
                                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                                            .addComponent(lastNameText)
                                            .addComponent(firstNameText, javax.swing.GroupLayout.PREFERRED_SIZE, 356, javax.swing.GroupLayout.PREFERRED_SIZE)))
                                    .addGroup(layout.createSequentialGroup()
                                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                            .addComponent(cityLabel)
                                            .addComponent(streetLabel))
                                        .addGap(18, 18, 18)
                                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                            .addComponent(streetText, javax.swing.GroupLayout.PREFERRED_SIZE, 597, javax.swing.GroupLayout.PREFERRED_SIZE)
                                            .addGroup(layout.createSequentialGroup()
                                                .addComponent(cityText, javax.swing.GroupLayout.PREFERRED_SIZE, 313, javax.swing.GroupLayout.PREFERRED_SIZE)
                                                .addGap(18, 18, 18)
                                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                                    .addComponent(nextButton, javax.swing.GroupLayout.PREFERRED_SIZE, 250, javax.swing.GroupLayout.PREFERRED_SIZE)
                                                    .addGroup(layout.createSequentialGroup()
                                                        .addComponent(stateLabel)
                                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                                        .addComponent(jComboBox1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))))))))))
                    .addGroup(layout.createSequentialGroup()
                        .addGap(23, 23, 23)
                        .addComponent(MainMenu, javax.swing.GroupLayout.PREFERRED_SIZE, 249, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(0, 0, Short.MAX_VALUE)))
                .addGap(20, 20, 20))
            .addGroup(layout.createSequentialGroup()
                .addGap(234, 234, 234)
                .addComponent(userInformationLabel)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGap(20, 20, 20)
                .addComponent(userInformationLabel)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(firstNameText, javax.swing.GroupLayout.PREFERRED_SIZE, 44, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(firstNameLabel))
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(lastNameLabel1)
                    .addComponent(lastNameText, javax.swing.GroupLayout.PREFERRED_SIZE, 44, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(streetText, javax.swing.GroupLayout.PREFERRED_SIZE, 44, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(streetLabel))
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(cityText, javax.swing.GroupLayout.PREFERRED_SIZE, 44, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cityLabel)
                    .addComponent(stateLabel)
                    .addComponent(jComboBox1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(zipText, javax.swing.GroupLayout.PREFERRED_SIZE, 44, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(zipLabel))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(MainMenu, javax.swing.GroupLayout.PREFERRED_SIZE, 65, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(nextButton, javax.swing.GroupLayout.PREFERRED_SIZE, 65, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap(42, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void zipTextActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_zipTextActionPerformed
    }//GEN-LAST:event_zipTextActionPerformed

    private void MainMenuActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_MainMenuActionPerformed
        
        if(firstNameText.getText().isEmpty() || firstNameText.getText()==null){
            Component frame = null;
                    JOptionPane.showMessageDialog(frame,
                    "Please enter your first name");   
        }
        else if(lastNameText.getText().isEmpty() || lastNameText.getText()==null){
            Component frame = null;
                    JOptionPane.showMessageDialog(frame,
                    "Please enter your last name");   
        }
        else if(streetText.getText().isEmpty() || streetText.getText()==null){
            Component frame = null;
                    JOptionPane.showMessageDialog(frame,
                    "Please enter your street address");   
        }
        else if(cityText.getText().isEmpty() || cityText.getText()==null){
            Component frame = null;
                    JOptionPane.showMessageDialog(frame,
                    "Please enter your city");   
        }
        else if(zipText.getText().length() != 5 || !zipText.getText().matches("[0-9]+")){
            Component frame = null;
                    JOptionPane.showMessageDialog(frame,
                    "Please enter your zipcode");   
        }
        else{
             if(jComboBox1.getSelectedItem() == null){
                Component frame = null;
                JOptionPane.showMessageDialog(frame,
                    "You have not selected your state");
            
            }
            else{
                start.userState =jComboBox1.getSelectedItem().toString();
                this.setVisible(false);
                new ControlPanelUser().setVisible(true);
            }
        
            start.userFirstName = firstNameText.getText();
            start.userLastName = lastNameText.getText();
            start.userStreet = streetText.getText();
            start.userCity = cityText.getText();
            start.userZip = zipText.getText();
        
            start.userFullName = start.userFirstName + " " + start.userLastName;
        
           
        }

    }//GEN-LAST:event_MainMenuActionPerformed

    private void nextButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_nextButtonActionPerformed
       if(firstNameText.getText().isEmpty() || firstNameText.getText()==null){
            Component frame = null;
                    JOptionPane.showMessageDialog(frame,
                    "Please enter your first name");   
        }
        else if(lastNameText.getText().isEmpty() || lastNameText.getText()==null){
            Component frame = null;
                    JOptionPane.showMessageDialog(frame,
                    "Please enter your last name");   
        }
        else if(streetText.getText().isEmpty() || streetText.getText()==null){
            Component frame = null;
                    JOptionPane.showMessageDialog(frame,
                    "Please enter your street address");   
        }
        else if(cityText.getText().isEmpty() || cityText.getText()==null){
            Component frame = null;
                    JOptionPane.showMessageDialog(frame,
                    "Please enter your city");   
        }
        else if(zipText.getText().length() != 5 || !zipText.getText().matches("[0-9]+")){
            Component frame = null;
                    JOptionPane.showMessageDialog(frame,
                    "Please enter your zipcode");   
        }
        else{
           
        
            start.userFirstName = firstNameText.getText();
            start.userLastName = lastNameText.getText();
            start.userStreet = streetText.getText();
            start.userCity = cityText.getText();
            start.userZip = zipText.getText();
        
            start.userFullName = start.userFirstName + " " + start.userLastName;
        
            if(jComboBox1.getSelectedItem() == null){
                Component frame = null;
                JOptionPane.showMessageDialog(frame,
                    "You have not selected your state");
                this.setVisible(false);
                new UserContacts().setVisible(true);
            }
            else{
                start.userState =jComboBox1.getSelectedItem().toString();
                this.setVisible(false);
                new ControlPanelUser().setVisible(true);
          //      start.userBool = false;
            }
        }
    }//GEN-LAST:event_nextButtonActionPerformed

    private void jComboBox1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jComboBox1ActionPerformed
    }//GEN-LAST:event_jComboBox1ActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(SetupControlPanel.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(SetupControlPanel.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(SetupControlPanel.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(SetupControlPanel.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new SetupControlPanel().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton MainMenu;
    private javax.swing.JTextField cityText;
    private javax.swing.JLabel firstNameLabel;
    private javax.swing.JTextField firstNameText;
    private javax.swing.JComboBox<String> jComboBox1;
    private javax.swing.JLabel lastNameLabel1;
    private javax.swing.JTextField lastNameText;
    private javax.swing.JButton nextButton;
    private javax.swing.JLabel streetLabel;
    private javax.swing.JTextField streetText;
    private javax.swing.JLabel userInformationLabel;
    private javax.swing.JTextField zipText;
    // End of variables declaration//GEN-END:variables
}
